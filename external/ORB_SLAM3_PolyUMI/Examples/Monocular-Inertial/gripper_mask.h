#ifndef POLYUMI_GRIPPER_MASK_H
#define POLYUMI_GRIPPER_MASK_H

#include <iostream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

// Blanks the gripper hardware that is rigidly attached to the camera -- fingers, their
// ArUco tags, the LED strips, the wiring PCB, the mirrors, and the body band -- before
// tracking.
//
// Those pixels sit at a fixed image location no matter where the camera is, so the ORB
// features they generate carry zero parallax (poisoning two-view init) and give every
// keyframe the same dominant DBoW2 signature (poisoning relocalization). Leaving them in
// cost us scene_2026-08-15_16-21-00_31d6: 498/506 failed two-view reconstructions while
// mapping, and 39 of 62 episodes that never relocalized at all.
//
// The mask is a PNG, not code, because its shape is a property of the physical gripper --
// hand-drawn against a temporal-median frame, where rigid hardware stays sharp and the
// scene blurs away. Upstream UMI does the same thing (Python rasterizes polygons to
// slam_mask.png, the binary consumes it), but its polygons describe *its* mount, and ours
// differs enough -- extra PCB, mirrors further outboard -- that porting the numbers
// misfit visibly. Path arrives via the settings YAML (see LoadSlamMask); ingest stamps it
// in the same way it stamps the atlas paths.
//
// Non-zero => discarded, zero => kept. Matches UMI, whose mask starts from np.zeros and
// draws the hardware in 255.

// Read Mask.Path from the already-open settings YAML and load it at tracking resolution.
// Returns an empty Mat when the key is absent, which disables masking -- callers must
// treat that as legal, since the out-of-tree binaries share this header.
inline cv::Mat LoadSlamMask(const cv::FileStorage &fsSettings, cv::Size img_size) {
  cv::FileNode node = fsSettings["Mask.Path"];
  if (node.empty()) {
    std::cout << "Mask.Path not set: tracking on unmasked frames" << std::endl;
    return cv::Mat();
  }

  const std::string path = static_cast<std::string>(node);
  cv::Mat mask = cv::imread(path, cv::IMREAD_GRAYSCALE);
  if (mask.empty()) {
    // Loud, but not fatal: a silently unmasked run looks fine until you read the logs a
    // day later and find nothing relocalized.
    std::cerr << "WARNING: Mask.Path set to '" << path
              << "' but it could not be read -- tracking on unmasked frames" << std::endl;
    return cv::Mat();
  }
  if (mask.size() != img_size) {
    std::cout << "Resizing mask from " << mask.size() << " to " << img_size << std::endl;
    // INTER_NEAREST keeps it binary; interpolating would produce a fringe of small
    // non-zero values, and any non-zero counts as masked.
    cv::resize(mask, mask, img_size, 0, 0, cv::INTER_NEAREST);
  }
  std::cout << "Loaded SLAM mask: " << path << " (" << mask.size() << ")" << std::endl;
  return mask;
}

// Apply a mask loaded by LoadSlamMask. A no-op when the mask is empty.
inline void ApplySlamMask(cv::Mat &img, const cv::Mat &mask) {
  if (mask.empty()) return;
  img.setTo(cv::Scalar(0, 0, 0), mask);
}

#endif  // POLYUMI_GRIPPER_MASK_H
