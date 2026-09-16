/**
 * This file is part of ORB-SLAM3
 *
 * Copyright (C) 2017-2020 Carlos Campos, Richard Elvira, Juan J. Gómez
 * Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 * Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós,
 * University of Zaragoza.
 *
 * ORB-SLAM3 is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * ORB-SLAM3. If not, see <http://www.gnu.org/licenses/>.
 */

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include <opencv2/core/core.hpp>

#include <System.h>

#include <json.h>

#include "gripper_mask.h"

using namespace std;
using nlohmann::json;
const double MS_TO_S = 1e-3; ///< Milliseconds to second conversion

bool LoadTelemetry(const string &strImuPath,
                   vector<double> &vTimeStamps,
                   vector<cv::Point3f> &vAcc,
                   vector<cv::Point3f> &vGyro);

int main(int argc, char **argv) {
  if (argc != 5 && argc != 6) {
    cerr << endl
         << "Usage: ./mono_inertial_gopro_vi path_to_vocabulary path_to_settings path_to_video path_to_telemetry [path_to_trajectory_output]"
         << endl;
    return 1;
  }

  vector<double> imuTimestamps;
  vector<cv::Point3f> vAcc, vGyr;
  if (!LoadTelemetry(argv[4], imuTimestamps, vAcc, vGyr)) {
    cerr << "Failed to load telemetry from: " << argv[4] << endl;
    return 1;
  }

  // open settings to get image resolution
  cv::FileStorage fsSettings(argv[2], cv::FileStorage::READ);
  if(!fsSettings.isOpened()) {
    cerr << "Failed to open settings file at: " << argv[2] << endl;
    return 1;
  }
  cv::Size img_size(fsSettings["Camera.width"],fsSettings["Camera.height"]);
  cv::Mat slam_mask = LoadSlamMask(fsSettings, img_size);
  bool bUseViewer = true;
  cv::FileNode fnViewer = fsSettings["System.Viewer"];
  if (!fnViewer.empty()) {
    bUseViewer = (int)fnViewer != 0;
  }
  fsSettings.release();

  // Create SLAM system. It initializes all system threads and gets ready to
  // process frames.
  ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::IMU_MONOCULAR, bUseViewer);

  // System ctor unconditionally clamps to QUIET; un-mute so IMU-init and
  // relocalization diagnostics reach our logs.
  ORB_SLAM3::Verbose::SetTh(ORB_SLAM3::Verbose::VERBOSITY_NORMAL);

  // Vector for tracking time statistics
  vector<float> vTimesTrack;
  cv::VideoCapture cap(argv[3]);
  // Check if camera opened successfully
  if (!cap.isOpened()) {
    std::cout << "Error opening video stream or file" << endl;
    return -1;
  }

  // Optional temporal decimation, matching mono_inertial_gopro_vi_localize's
  // POLYUMI_SLAM_FRAME_STRIDE: keep every Nth frame (1 = every frame = the
  // original behaviour, bit-identical when the variable is unset).  Divide
  // Camera.fps in the settings YAML by the same N, since ORB-SLAM3 derives its
  // keyframe-insertion window (mMaxFrames) from it.
  int frame_stride = 1;
  if (const char *env_stride = std::getenv("POLYUMI_SLAM_FRAME_STRIDE")) {
    const int parsed = std::atoi(env_stride);
    if (parsed > 0) frame_stride = parsed;
    else
      cerr << "Ignoring invalid POLYUMI_SLAM_FRAME_STRIDE=" << env_stride
           << " (want a positive integer)" << endl;
  }

  // Main loop
  int cnt_empty_frame = 0;
  int img_id = 0;
  // CAP_PROP_FPS reports 0 (or NaN) for containers OpenCV can't get a rate out of.
  // Unguarded that makes frame_diff_s inf/NaN, and the usleep() below then either
  // hangs the run or is skipped entirely -- so fall back to the GoPro's nominal 60.
  double fps = cap.get(cv::CAP_PROP_FPS);
  if (!(fps > 1e-6)) {  // NaN-safe: a NaN fps fails this and takes the fallback
    cerr << "CAP_PROP_FPS reported " << fps << "; falling back to 60 fps for loop pacing"
         << endl;
    fps = 60.0;
  }
  // Spacing between the frames we actually feed, not between source frames --
  // this paces the loop against wall clock below, and decimating widens the
  // real interval by the stride.  Getting this wrong would under-sleep and
  // starve LocalMapping of the background time the throttle exists to give it.
  double frame_diff_s = frame_stride / fps;
  if (frame_stride != 1)
    std::cout << "Frame stride " << frame_stride << ": feeding every " << frame_stride
              << "th frame (effective " << (fps / frame_stride) << " fps)\n";
  std::vector<ORB_SLAM3::IMU::Point> vImuMeas;
  size_t last_imu_idx = 0;
  while (1) {
    cv::Mat im,im_track;
    bool success = cap.read(im);

    if (!success) {
      cnt_empty_frame++;
      std::cout<<"Empty frame...\n";
      if (cnt_empty_frame > 100)
        break;
      continue;
    }
      im_track = im.clone();
      double tframe = cap.get(cv::CAP_PROP_POS_MSEC) * MS_TO_S;
      ++img_id;

      cv::resize(im_track, im_track, img_size);
      ApplySlamMask(im_track, slam_mask);

      // gather imu measurements between frames
      // Load imu measurements from previous frame
      vImuMeas.clear();
      while(last_imu_idx < imuTimestamps.size() && imuTimestamps[last_imu_idx] <= tframe && tframe > 0)
      {
          vImuMeas.push_back(ORB_SLAM3::IMU::Point(vAcc[last_imu_idx].x,vAcc[last_imu_idx].y,vAcc[last_imu_idx].z,
                                                   vGyr[last_imu_idx].x,vGyr[last_imu_idx].y,vGyr[last_imu_idx].z,
                                                   imuTimestamps[last_imu_idx]));
          last_imu_idx++;
      }


      auto t1 = std::chrono::steady_clock::now();
      SLAM.TrackMonocular(im_track, tframe, vImuMeas);
      auto t2 = std::chrono::steady_clock::now();
      double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();

      if (img_id % 100 == 0) {
        std::cout<<"Video FPS: "<<1./frame_diff_s<<"\n";
        std::cout<<"ORB-SLAM 3 running at: "<<1./ttrack<< " FPS\n";
      }
      vTimesTrack.push_back(ttrack);

      // Wait to load the next frame
      if (ttrack < frame_diff_s)
        usleep((frame_diff_s - ttrack) * 1e6);

      // Advance past the frames we're dropping.  grab() still decodes (HEVC
      // inter-frame deps demand it) but skips retrieve()'s colour conversion
      // and Mat copy.
      for (int s = 1; s < frame_stride; ++s) {
        if (!cap.grab()) break;
      }
  }

  if (bUseViewer) {
    cout << "\nSLAM complete. Press Enter to close the viewer..." << endl;
    cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }

  // Stop all threads
  SLAM.Shutdown();

  // NOTE: do not call PrintLoadedAtlasState here. Shutdown() returns before
  // LocalMapping/LoopClosing fully release their map mutexes (the stdout still
  // prints "mpLocalMapper is not finished" / "mpLoopCloser is not finished"),
  // and our diagnostic acquires Atlas + Map locks, which deadlocks.  The
  // localizer logs the loaded state instead, which is what we actually need
  // to debug relocalization.

  // Tracking time statistics
  if (!vTimesTrack.empty()) {
    sort(vTimesTrack.begin(), vTimesTrack.end());
    float totaltime = 0;
    for (float t : vTimesTrack) totaltime += t;
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[vTimesTrack.size() / 2] << endl;
    cout << "mean tracking time: " << totaltime / vTimesTrack.size() << endl;
  }

  // Save camera trajectory
  SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

  if (argc == 6) {
    cout << "Saving CSV trajectory to: " << argv[5] << endl;
    // Camera-frame CSV, matching the localizer -- see mono_inertial_gopro_vi_localize.cc.
    SLAM.SaveTrajectoryCSV(argv[5]);
  }

  return 0;
}

bool LoadTelemetry(const string &path_to_telemetry_file,
                   vector<double> &vTimeStamps,
                   vector<cv::Point3f> &vAcc,
                   vector<cv::Point3f> &vGyro) {

    std::ifstream file;
    file.open(path_to_telemetry_file.c_str());
    if (!file.is_open()) {
      return false;
    }
    json j;
    file >> j;
    const auto accl = j["1"]["streams"]["ACCL"]["samples"];
    const auto gyro = j["1"]["streams"]["GYRO"]["samples"];
    const auto gps5 = j["1"]["streams"]["GPS5"]["samples"];
    std::map<double, cv::Point3f> sorted_acc;
    std::map<double, cv::Point3f> sorted_gyr;

    for (const auto &e : accl) {
      cv::Point3f v((float)e["value"][1], (float)e["value"][2], (float)e["value"][0]);
      sorted_acc.insert(std::make_pair((double)e["cts"] * MS_TO_S, v));
    }
    for (const auto &e : gyro) {
      cv::Point3f v((float)e["value"][1], (float)e["value"][2], (float)e["value"][0]);
      sorted_gyr.insert(std::make_pair((double)e["cts"] * MS_TO_S, v));
    }
//    for (const auto &e : gps5) {
//      Eigen::Vector3d v;
//      Eigen::Vector2d vel2d_vel3d;
//      v << e["value"][0], e["value"][1], e["value"][2];
//      vel2d_vel3d << e["value"][3], e["value"][4];
//      telemetry.gps.lle.emplace_back(v);
//      telemetry.gps.timestamp_ms.emplace_back(e["cts"]);
//      telemetry.gps.precision.emplace_back(e["precision"]);
//      telemetry.gps.vel2d_vel3d.emplace_back(vel2d_vel3d);
//    }

    double imu_start_t = sorted_acc.begin()->first;
    for (auto acc : sorted_acc) {
        vTimeStamps.push_back(acc.first-imu_start_t);
        vAcc.push_back(acc.second);
    }
    for (auto gyr : sorted_gyr) {
        vGyro.push_back(gyr.second);
    }
    file.close();
    return true;
}
