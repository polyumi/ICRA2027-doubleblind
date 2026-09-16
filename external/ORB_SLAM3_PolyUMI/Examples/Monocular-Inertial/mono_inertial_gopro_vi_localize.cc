/**
 * Localization mode counterpart to mono_inertial_gopro_vi.
 *
 * Loads a pre-built atlas (via System.LoadAtlasFromFile in the settings YAML)
 * and runs ORB-SLAM3 in localization-only mode against a GoPro mp4 + telemetry
 * JSON. Writes the per-frame trajectory as CSV (SaveTrajectoryCSV: one row per
 * frame fed, with frame_idx and an explicit is_lost column) to a caller-specified
 * path so downstream tooling can pick it up without scraping cwd.
 *
 * Usage:
 *   ./mono_inertial_gopro_vi_localize \
 *       path_to_vocabulary \
 *       path_to_settings \
 *       path_to_video \
 *       path_to_telemetry \
 *       path_to_trajectory_output \
 *       [path_to_reverse_trajectory_output]
 *
 * With five arguments (the production path — this is what slam_step.py invokes)
 * a single forward pass runs and frames are streamed straight off the decoder,
 * one at a time.  Do not "optimize" that into a decode-everything-first loop: at
 * the calibrated 1352x1014 a frame is ~4.1 MB, so a minute of video at stride 2
 * is several GB of resident memory for no benefit.
 *
 * With the optional 6th argument, the video is instead decoded once into memory
 * and localized twice against the same loaded atlas: a normal forward pass
 * (written to path_to_trajectory_output) and a second reverse-order pass
 * (written to path_to_reverse_trajectory_output).  Buffering earns its cost only
 * here, where the frames must be walked twice.  Because the run is offline, the
 * reverse pass can recover the lead-in frames the forward pass loses before it
 * manages to relocalize — it reaches them from the well-tracked middle with a
 * motion prior instead of a cold relocalization.  Its trajectory is on a flipped
 * clock (timestamp = t_last - t_video); slam_step.py maps it back and merges.
 *
 * The settings YAML must contain `System.LoadAtlasFromFile: <atlas_path>`.
 * PolyUMI's slam_step.py injects this into a temp copy of the YAML before
 * invoking this binary.
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
const double MS_TO_S = 1e-3;

bool LoadTelemetry(const string &strImuPath,
                   vector<double> &vTimeStamps,
                   vector<cv::Point3f> &vAcc,
                   vector<cv::Point3f> &vGyro);

// Sanitizes CAP_PROP_POS_MSEC into a strictly increasing clock.  The FFmpeg backend
// often reports 0 once the demuxer hits EOF, which would poison ORB-SLAM3's "timestamp
// older than previous" guard and, in two-pass mode, the reverse pass's anchor (the last
// frame time).  Any non-monotonic reading is extrapolated from the last good one; the
// frame itself is always kept, only the timestamp is repaired.
struct MonotonicVideoClock {
  double dt_fallback;
  bool has_last = false;
  double last = 0.0;
  int n_fixed = 0;

  // dt_fallback is scaled by the stride because it extrapolates between *kept* frames,
  // which sit frame_stride source frames apart.
  MonotonicVideoClock(double fps, int frame_stride)
      : dt_fallback(((fps > 1e-6) ? (1.0 / fps) : (1.0 / 60.0)) * frame_stride) {}

  double Next(double raw) {
    double t = raw;
    if (has_last && t <= last) {
      t = last + dt_fallback;
      n_fixed++;
    }
    has_last = true;
    last = t;
    return t;
  }

  void Report() const {
    if (n_fixed > 0)
      cout << "Repaired " << n_fixed << " non-monotonic frame timestamp(s) via "
           << dt_fallback << "s fallback spacing" << endl;
  }
};

// One localization sweep against a freshly loaded atlas.  Frames are handed over one at
// a time in increasing-timestamp order, whatever the caller's frame source; the
// trajectory is written by Finish().  A fresh System per sweep is what gives the reverse
// pass a clean tracker that starts LOST and relocalizes into the loaded map, without
// ORB-SLAM3 needing a "reset-tracking-but-keep-map" entry point.
class LocalizationPass {
 public:
  LocalizationPass(const char *vocab, const char *settings,
                   const vector<double> &imuTs, const vector<cv::Point3f> &acc,
                   const vector<cv::Point3f> &gyr, const char *tag)
      // LoadAtlasFromFile in the YAML triggers atlas restoration in the ctor; the
      // Cheng fork already ChangeMap()s the loaded map active, so no restore call.
      : SLAM_(vocab, settings, ORB_SLAM3::System::IMU_MONOCULAR, false),
        imuTs_(imuTs), acc_(acc), gyr_(gyr), tag_(tag) {
    // The ctor clamps verbosity to QUIET; re-enable reloc/lost/IMU-init messages.
    ORB_SLAM3::Verbose::SetTh(ORB_SLAM3::Verbose::VERBOSITY_NORMAL);
    // Tracking-only: don't grow the loaded map. Must come after construction.
    SLAM_.ActivateLocalizationMode();
    SLAM_.PrintLoadedAtlasState(tag);
  }

  // Track one frame, first draining every IMU sample that precedes it.
  void Feed(const cv::Mat &im, double tframe) {
    vImuMeas_.clear();
    while (imu_idx_ < imuTs_.size() && imuTs_[imu_idx_] <= tframe && tframe > 0) {
      vImuMeas_.push_back(ORB_SLAM3::IMU::Point(
          acc_[imu_idx_].x, acc_[imu_idx_].y, acc_[imu_idx_].z, gyr_[imu_idx_].x,
          gyr_[imu_idx_].y, gyr_[imu_idx_].z, imuTs_[imu_idx_]));
      imu_idx_++;
    }
    SLAM_.TrackMonocular(im, tframe, vImuMeas_);
    if (++n_fed_ % 100 == 0)
      cout << tag_ << " tracked " << n_fed_ << " frames" << endl;
  }

  size_t n_fed() const { return n_fed_; }

  void Finish(const string &traj_out) {
    SLAM_.Shutdown();
    cout << "Saving trajectory to: " << traj_out << endl;
    // CSV, not EuRoC: SaveTrajectoryEuRoC's inertial branch composes mImuCalib.mTbc and so
    // reports the *IMU body* pose, while SaveTrajectoryCSV reports the camera pose in the
    // optical frame -- which is the frame the policy's observations live in, and the one
    // upstream UMI trains against. The CSV also carries frame_idx and an explicit is_lost
    // column, so the Python side can index rows directly instead of matching timestamps.
    SLAM_.SaveTrajectoryCSV(traj_out);
  }

 private:
  ORB_SLAM3::System SLAM_;
  const vector<double> &imuTs_;
  const vector<cv::Point3f> &acc_;
  const vector<cv::Point3f> &gyr_;
  const char *tag_;
  size_t imu_idx_ = 0;
  size_t n_fed_ = 0;
  std::vector<ORB_SLAM3::IMU::Point> vImuMeas_;
};

// Advance the capture past the frames a stride > 1 drops.  grab() still decodes (HEVC
// inter-frame deps demand it) but skips retrieve()'s colour conversion and Mat copy.
static void SkipStrideFrames(cv::VideoCapture &cap, int frame_stride) {
  for (int s = 1; s < frame_stride; ++s) {
    if (!cap.grab()) break;
  }
}

// Forward pass straight off the decoder, one frame in flight at a time.
//
// This is the production path: slam_step.py passes five arguments, never the sixth, so
// the reverse pass is not requested.  Streaming rather than buffering matters because a
// frame at the calibrated 1352x1014 is ~4.1 MB, so holding a whole episode in memory
// costs gigabytes -- a cost worth paying only for the two-pass mode that actually needs
// to walk the frames twice.  Returns frames fed, or -1 on error.
static long StreamForwardPass(const char *vocab, const char *settings,
                              const char *video, cv::Size img_size,
                              int frame_stride, const vector<double> &imuTs,
                              const vector<cv::Point3f> &acc,
                              const vector<cv::Point3f> &gyr,
                              const string &traj_out, const cv::Mat &slam_mask) {
  cv::VideoCapture cap(video);
  if (!cap.isOpened()) {
    cerr << "Error opening video stream or file: " << video << endl;
    return -1;
  }
  MonotonicVideoClock clock(cap.get(cv::CAP_PROP_FPS), frame_stride);

  LocalizationPass pass(vocab, settings, imuTs, acc, gyr,
                        "[localizer:forward] post-load");

  int cnt_empty_frame = 0;
  cv::Mat im, resized;
  while (true) {
    if (!cap.read(im)) {
      if (++cnt_empty_frame > 100) break;
      continue;
    }
    cnt_empty_frame = 0;
    const double tframe = clock.Next(cap.get(cv::CAP_PROP_POS_MSEC) * MS_TO_S);
    cv::resize(im, resized, img_size);
    ApplySlamMask(resized, slam_mask);
    pass.Feed(resized, tframe);
    SkipStrideFrames(cap, frame_stride);
  }
  clock.Report();

  if (pass.n_fed() < 2) {
    cerr << "Decoded fewer than 2 frames from " << video << endl;
    return -1;
  }
  cout << "Streamed " << pass.n_fed() << " frames" << endl;
  pass.Finish(traj_out);
  return static_cast<long>(pass.n_fed());
}

// Decode + resize every kept frame into memory.  Only used by the two-pass mode, where
// the reverse sweep must revisit the same frames and re-decoding (or an ffmpeg
// re-encode) would cost more than the RAM.
static bool DecodeAllFrames(const char *video, cv::Size img_size, int frame_stride,
                            vector<cv::Mat> &frames, vector<double> &frameTimes,
                            const cv::Mat &slam_mask) {
  cv::VideoCapture cap(video);
  if (!cap.isOpened()) {
    cerr << "Error opening video stream or file: " << video << endl;
    return false;
  }
  MonotonicVideoClock clock(cap.get(cv::CAP_PROP_FPS), frame_stride);

  int cnt_empty_frame = 0;
  while (true) {
    cv::Mat im;
    if (!cap.read(im)) {
      if (++cnt_empty_frame > 100) break;
      continue;
    }
    cnt_empty_frame = 0;
    const double tframe = clock.Next(cap.get(cv::CAP_PROP_POS_MSEC) * MS_TO_S);
    cv::Mat resized;
    cv::resize(im, resized, img_size);
    ApplySlamMask(resized, slam_mask);
    frames.push_back(std::move(resized));
    frameTimes.push_back(tframe);
    SkipStrideFrames(cap, frame_stride);
  }
  clock.Report();
  return true;
}

// Walk a decoded buffer through one pass.  ``reversed`` goes back-to-front on a flipped
// clock (t' = t_last - t) so timestamps still increase; the caller passes IMU arrays
// already transformed to match (see main()).
static void FeedBuffer(LocalizationPass &pass, const vector<cv::Mat> &frames,
                       const vector<double> &frameTimes, bool reversed) {
  const size_t N = frames.size();
  const double t_last = frameTimes[N - 1];
  for (size_t k = 0; k < N; ++k) {
    const size_t j = reversed ? (N - 1 - k) : k;  // original frame index
    pass.Feed(frames[j], reversed ? (t_last - frameTimes[j]) : frameTimes[j]);
  }
}

int main(int argc, char **argv) {
  if (argc != 6 && argc != 7) {
    cerr << endl
         << "Usage: ./mono_inertial_gopro_vi_localize path_to_vocabulary "
            "path_to_settings path_to_video path_to_telemetry "
            "path_to_trajectory_output [path_to_reverse_trajectory_output]"
         << endl
         << "  With five args, one forward pass runs and frames are streamed off "
            "the decoder (peak memory: one frame).\n"
            "  With the optional 6th arg, a second reverse-order pass is run "
            "against the same atlas and its trajectory written there; this "
            "offline back-pass recovers lead-in frames the forward pass loses "
            "before it can relocalize.  That mode decodes the whole video into "
            "memory once, since both passes walk the same frames."
         << endl;
    return 1;
  }

  const string traj_out = argv[5];
  const bool do_reverse = (argc == 7);
  const string traj_out_reverse = do_reverse ? argv[6] : "";

  vector<double> imuTimestamps;
  vector<cv::Point3f> vAcc, vGyr;
  if (!LoadTelemetry(argv[4], imuTimestamps, vAcc, vGyr)) {
    cerr << "Failed to load telemetry from: " << argv[4] << endl;
    return 1;
  }

  cv::FileStorage fsSettings(argv[2], cv::FileStorage::READ);
  if (!fsSettings.isOpened()) {
    cerr << "Failed to open settings file at: " << argv[2] << endl;
    return 1;
  }
  cv::Size img_size(fsSettings["Camera.width"], fsSettings["Camera.height"]);
  cv::Mat slam_mask = LoadSlamMask(fsSettings, img_size);
  fsSettings.release();

  // Optional temporal decimation, for tuning experiments only: keep every Nth
  // decoded frame (1 = every frame = production behaviour).  Env var rather than
  // a CLI arg so the argv contract stays as documented above and an unset
  // environment is bit-identical to the pre-existing code path.  Note the
  // settings YAML's Camera.fps should be divided by the same N when using this,
  // since ORB-SLAM3 derives its keyframe-insertion window (mMaxFrames) from it.
  int frame_stride = 1;
  if (const char *env_stride = std::getenv("POLYUMI_SLAM_FRAME_STRIDE")) {
    const int parsed = std::atoi(env_stride);
    if (parsed > 0) frame_stride = parsed;
    else
      cerr << "Ignoring invalid POLYUMI_SLAM_FRAME_STRIDE=" << env_stride
           << " (want a positive integer)" << endl;
  }

  if (frame_stride != 1)
    cout << "Frame stride " << frame_stride << ": feeding every " << frame_stride
         << "th source frame" << endl;

  // Single-pass (production) path: stream frames straight off the decoder, so peak
  // memory is one frame rather than the whole episode.  Nothing here needs to revisit a
  // frame, and at ~4.1 MB per decoded frame the buffer below is gigabytes on a long
  // episode.
  if (!do_reverse) {
    return StreamForwardPass(argv[1], argv[2], argv[3], img_size, frame_stride,
                             imuTimestamps, vAcc, vGyr, traj_out, slam_mask) < 0
               ? 1
               : 0;
  }

  // Two-pass path: the reverse sweep walks the same frames a second time, so decode
  // them once into memory rather than re-decoding.  Frames are stored at tracking
  // resolution (img_size), which is the only thing bounding the memory here.
  vector<cv::Mat> frames;
  vector<double> frameTimes;
  if (!DecodeAllFrames(argv[3], img_size, frame_stride, frames, frameTimes, slam_mask)) return 1;
  if (frames.size() < 2) {
    cerr << "Decoded fewer than 2 frames from " << argv[3] << endl;
    return 1;
  }
  cout << "Decoded " << frames.size() << " frames into memory" << endl;

  {
    LocalizationPass forward(argv[1], argv[2], imuTimestamps, vAcc, vGyr,
                             "[localizer:forward] post-load");
    FeedBuffer(forward, frames, frameTimes, /*reversed=*/false);
    forward.Finish(traj_out);
  }

  // Time-reversed IMU: reverse sample order, negate gyro (angular velocity flips under
  // t -> -t), keep accelerometer (specific force is invariant), and remap timestamps
  // onto the flipped clock (t' = t_anchor - t) so they still increase.  Mirrors the
  // reversal validated in the Python prototype.
  //
  // The anchor MUST be the same reference FeedBuffer uses to flip the frame clock (the
  // last frame time), NOT the last IMU time.  The IMU stream (200 Hz) outlasts the video
  // (60 Hz), so anchoring the IMU at its own end would shift the whole reversed
  // IMU/frame alignment by (imu_end - vid_end).  Samples past the last frame get a
  // negative t' and are consumed at the first reversed frame with a positive timestamp,
  // which is where they belong temporally.
  const double t_anchor = frameTimes.back();
  vector<double> imuTsRev;
  vector<cv::Point3f> vAccRev, vGyrRev;
  imuTsRev.reserve(imuTimestamps.size());
  vAccRev.reserve(vAcc.size());
  vGyrRev.reserve(vGyr.size());
  for (size_t i = imuTimestamps.size(); i-- > 0;) {
    imuTsRev.push_back(t_anchor - imuTimestamps[i]);
    vAccRev.push_back(vAcc[i]);
    vGyrRev.push_back(cv::Point3f(-vGyr[i].x, -vGyr[i].y, -vGyr[i].z));
  }
  {
    LocalizationPass reverse(argv[1], argv[2], imuTsRev, vAccRev, vGyrRev,
                             "[localizer:reverse] post-load");
    FeedBuffer(reverse, frames, frameTimes, /*reversed=*/true);
    reverse.Finish(traj_out_reverse);
  }

  return 0;
}

bool LoadTelemetry(const string &path_to_telemetry_file,
                   vector<double> &vTimeStamps,
                   vector<cv::Point3f> &vAcc,
                   vector<cv::Point3f> &vGyro) {
  std::ifstream file(path_to_telemetry_file.c_str());
  if (!file.is_open()) return false;

  json j;
  file >> j;
  const auto accl = j["1"]["streams"]["ACCL"]["samples"];
  const auto gyro = j["1"]["streams"]["GYRO"]["samples"];

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

  double imu_start_t = sorted_acc.begin()->first;
  for (auto acc : sorted_acc) {
    vTimeStamps.push_back(acc.first - imu_start_t);
    vAcc.push_back(acc.second);
  }
  for (auto gyr : sorted_gyr) {
    vGyro.push_back(gyr.second);
  }
  return true;
}
