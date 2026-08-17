#include "ManusSDK.h"
#include "ManusSDKTypeInitializers.h"

#include <algorithm>
#include <atomic>
#include <array>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace {
constexpr std::size_t kMaxNodes = 64;
constexpr std::size_t kHandErgonomicsCount = 20;

struct RawFrame {
    uint64_t sequence = 0;
    uint64_t publish_time = 0;
    uint32_t glove_id = 0;
    std::vector<SkeletonNode> nodes;
};

struct ErgonomicsFrame {
    uint32_t id = 0;
    bool is_user_id = false;
    std::array<float, ErgonomicsDataType_MAX_SIZE> data{};
};

std::mutex g_mutex;
std::vector<RawFrame> g_latest_frames;
std::vector<ErgonomicsFrame> g_latest_ergonomics;
uint64_t g_sequence = 0;
bool g_initialized = false;
std::atomic_bool g_connected{false};
bool g_pinch_compensation = false;
std::string g_last_error;

void SetError(const char* operation, SDKReturnCode code) {
    g_last_error = std::string(operation) + " failed (SDKReturnCode=" +
                   std::to_string(static_cast<int>(code)) + ")";
}

bool ApplyRawSkeletonSettings() {
    SDKReturnCode code = CoreSdk_SetRawSkeletonPinchCompensation(
        g_pinch_compensation);
    if (code != SDKReturnCode_Success) {
        SetError("CoreSdk_SetRawSkeletonPinchCompensation", code);
        return false;
    }
    bool enabled = false;
    code = CoreSdk_GetRawSkeletonPinchCompensation(&enabled);
    if (code != SDKReturnCode_Success) {
        SetError("CoreSdk_GetRawSkeletonPinchCompensation", code);
        return false;
    }
    if (enabled != g_pinch_compensation) {
        g_last_error = "Raw Skeleton pinch compensation readback mismatch";
        return false;
    }
    return true;
}

void OnRawSkeletonStream(const SkeletonStreamInfo* const stream) {
    if (stream == nullptr || stream->skeletonsCount == 0) return;

    std::vector<RawFrame> collection;
    collection.reserve(stream->skeletonsCount);
    for (uint32_t index = 0; index < stream->skeletonsCount; ++index) {
        RawSkeletonInfo info;
        RawSkeletonInfo_Init(&info);
        if (CoreSdk_GetRawSkeletonInfo(index, &info) != SDKReturnCode_Success ||
            info.nodesCount == 0 || info.nodesCount > kMaxNodes) {
            continue;
        }
        std::vector<SkeletonNode> nodes(info.nodesCount);
        if (CoreSdk_GetRawSkeletonData(index, nodes.data(), info.nodesCount) !=
            SDKReturnCode_Success) {
            continue;
        }

        RawFrame frame;
        frame.publish_time = stream->publishTime.time;
        frame.glove_id = info.gloveId;
        frame.nodes = std::move(nodes);
        collection.push_back(std::move(frame));
    }
    if (collection.empty()) return;

    std::lock_guard<std::mutex> lock(g_mutex);
    ++g_sequence;
    for (RawFrame& frame : collection) frame.sequence = g_sequence;
    g_latest_frames = std::move(collection);
}

void OnErgonomicsStream(const ErgonomicsStream* const stream) {
    if (stream == nullptr || stream->dataCount == 0) return;
    std::vector<ErgonomicsFrame> collection;
    collection.reserve(stream->dataCount);
    for (uint32_t index = 0; index < stream->dataCount; ++index) {
        const ErgonomicsData& source = stream->data[index];
        ErgonomicsFrame frame;
        frame.id = source.id;
        frame.is_user_id = source.isUserID;
        std::copy(std::begin(source.data), std::end(source.data), frame.data.begin());
        collection.push_back(frame);
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    g_latest_ergonomics = std::move(collection);
}

GloveCalibrationArgs CalibrationArgs(uint32_t glove_id) {
    GloveCalibrationArgs args;
    GloveCalibrationArgs_Init(&args);
    args.gloveId = glove_id;
    return args;
}

GloveCalibrationStepArgs CalibrationStepArgs(uint32_t glove_id, uint32_t step) {
    GloveCalibrationStepArgs args;
    GloveCalibrationStepArgs_Init(&args);
    args.gloveId = glove_id;
    args.stepIndex = step;
    return args;
}

int CalibrationResult(const char* operation, SDKReturnCode code, bool result) {
    if (code != SDKReturnCode_Success) {
        SetError(operation, code);
        return -1;
    }
    if (!result) {
        g_last_error = std::string(operation) + " was rejected by MANUS Core";
        return -1;
    }
    g_last_error.clear();
    return 0;
}
}  // namespace

extern "C" {

struct ManusBridgeNode {
    uint32_t node_id;
    uint32_t parent_id;
    int32_t side;
    int32_t chain_type;
    int32_t finger_joint_type;
    float position[3];
    // Exact official ManusQuaternion field order: w, x, y, z.
    float rotation_wxyz[4];
};

struct ManusBridgeFrame {
    uint64_t sequence;
    uint64_t publish_time;
    uint32_t glove_id;
    uint32_t node_count;
    int32_t side;
    ManusBridgeNode nodes[kMaxNodes];
    int32_t has_ergonomics;
    float ergonomics[kHandErgonomicsCount];
};

struct ManusBridgeCalibrationStep {
    uint32_t index;
    char title[MAX_NUM_CHARS_IN_CALIBRATION_TITLE];
    char description[MAX_NUM_CHARS_IN_CALIBRATION_DESCRIPTION];
    float time;
};

int manus_bridge_initialize() {
    if (g_initialized) return 0;
    SDKReturnCode code = CoreSdk_InitializeIntegrated();
    if (code != SDKReturnCode_Success) {
        SetError("CoreSdk_InitializeIntegrated", code);
        return -1;
    }
    code = CoreSdk_RegisterCallbackForRawSkeletonStream(OnRawSkeletonStream);
    if (code != SDKReturnCode_Success) {
        SetError("CoreSdk_RegisterCallbackForRawSkeletonStream", code);
        CoreSdk_ShutDown();
        return -1;
    }
    code = CoreSdk_RegisterCallbackForErgonomicsStream(OnErgonomicsStream);
    if (code != SDKReturnCode_Success) {
        SetError("CoreSdk_RegisterCallbackForErgonomicsStream", code);
        CoreSdk_ShutDown();
        return -1;
    }

    CoordinateSystemVUH coordinate;
    CoordinateSystemVUH_Init(&coordinate);
    coordinate.handedness = Side_Right;
    coordinate.up = AxisPolarity_PositiveZ;
    coordinate.view = AxisView_XFromViewer;
    coordinate.unitScale = 1.0f;  // metres
    constexpr bool kUseWorldCoordinates = true;
    code = CoreSdk_InitializeCoordinateSystemWithVUH(
        coordinate, kUseWorldCoordinates);
    if (code != SDKReturnCode_Success) {
        SetError("CoreSdk_InitializeCoordinateSystemWithVUH", code);
        CoreSdk_ShutDown();
        return -1;
    }
    g_initialized = true;
    g_last_error.clear();
    return 0;
}

int manus_bridge_connect() {
    if (!g_initialized) return -1;
    bool connected = false;
    if (CoreSdk_GetIsConnectedToCore(&connected) == SDKReturnCode_Success &&
        connected) {
        if (!ApplyRawSkeletonSettings()) return -1;
        g_connected = true;
        return 0;
    }
    g_connected = false;

    SDKReturnCode code = CoreSdk_LookForHosts(1, false);
    if (code != SDKReturnCode_Success) {
        SetError("CoreSdk_LookForHosts", code);
        return -1;
    }
    uint32_t count = 0;
    code = CoreSdk_GetNumberOfAvailableHostsFound(&count);
    if (code != SDKReturnCode_Success || count == 0) {
        if (code != SDKReturnCode_Success) SetError("CoreSdk_GetNumberOfAvailableHostsFound", code);
        else g_last_error = "No MANUS Core/Integrated host found";
        return -1;
    }
    std::vector<ManusHost> hosts(count);
    code = CoreSdk_GetAvailableHostsFound(hosts.data(), count);
    if (code != SDKReturnCode_Success) {
        SetError("CoreSdk_GetAvailableHostsFound", code);
        return -1;
    }
    code = CoreSdk_ConnectToHost(hosts.front());
    if (code != SDKReturnCode_Success) {
        SetError("CoreSdk_ConnectToHost", code);
        return -1;
    }
    if (!ApplyRawSkeletonSettings()) return -1;
    // WORLD positions should move using the best available MANUS tracking input.
    CoreSdk_SetRawSkeletonHandMotion(HandMotion_Auto);
    g_connected = true;
    g_last_error.clear();
    return 0;
}

int manus_bridge_set_pinch_compensation(int enabled) {
    g_pinch_compensation = enabled != 0;
    return 0;
}

int manus_bridge_get_pinch_compensation(int* enabled) {
    if (enabled == nullptr || !g_connected.load()) return -1;
    bool value = false;
    SDKReturnCode code = CoreSdk_GetRawSkeletonPinchCompensation(&value);
    if (code != SDKReturnCode_Success) {
        SetError("CoreSdk_GetRawSkeletonPinchCompensation", code);
        return -1;
    }
    *enabled = value ? 1 : 0;
    return 0;
}

int manus_bridge_calibration_get_step_count(uint32_t glove_id, uint32_t* count) {
    if (!g_connected.load() || count == nullptr) return -1;
    const SDKReturnCode code = CoreSdk_GloveCalibrationGetNumberOfSteps(
        CalibrationArgs(glove_id), count);
    if (code != SDKReturnCode_Success) {
        SetError("CoreSdk_GloveCalibrationGetNumberOfSteps", code);
        return -1;
    }
    if (*count == 0) {
        g_last_error = "MANUS returned an empty glove calibration sequence";
        return -1;
    }
    g_last_error.clear();
    return 0;
}

int manus_bridge_calibration_get_step(
    uint32_t glove_id, uint32_t step, ManusBridgeCalibrationStep* output) {
    if (!g_connected.load() || output == nullptr) return -1;
    GloveCalibrationStepData data;
    GloveCalibrationStepData_Init(&data);
    const SDKReturnCode code = CoreSdk_GloveCalibrationGetStepData(
        CalibrationStepArgs(glove_id, step), &data);
    if (code != SDKReturnCode_Success) {
        SetError("CoreSdk_GloveCalibrationGetStepData", code);
        return -1;
    }
    std::memset(output, 0, sizeof(*output));
    output->index = data.index;
    std::copy_n(data.title, sizeof(output->title), output->title);
    std::copy_n(data.description, sizeof(output->description), output->description);
    output->time = data.time;
    g_last_error.clear();
    return 0;
}

int manus_bridge_calibration_start(uint32_t glove_id) {
    bool result = false;
    const SDKReturnCode code = CoreSdk_GloveCalibrationStart(
        CalibrationArgs(glove_id), &result);
    return CalibrationResult("CoreSdk_GloveCalibrationStart", code, result);
}

int manus_bridge_calibration_run_step(uint32_t glove_id, uint32_t step) {
    bool result = false;
    const SDKReturnCode code = CoreSdk_GloveCalibrationStartStep(
        CalibrationStepArgs(glove_id, step), &result);
    return CalibrationResult("CoreSdk_GloveCalibrationStartStep", code, result);
}

int manus_bridge_calibration_finish(uint32_t glove_id) {
    bool result = false;
    const SDKReturnCode code = CoreSdk_GloveCalibrationFinish(
        CalibrationArgs(glove_id), &result);
    return CalibrationResult("CoreSdk_GloveCalibrationFinish", code, result);
}

int manus_bridge_calibration_stop(uint32_t glove_id) {
    bool result = false;
    const SDKReturnCode code = CoreSdk_GloveCalibrationStop(
        CalibrationArgs(glove_id), &result);
    return CalibrationResult("CoreSdk_GloveCalibrationStop", code, result);
}

int manus_bridge_calibration_export(
    uint32_t glove_id, unsigned char* output, uint32_t capacity,
    uint32_t* required_size) {
    if (!g_connected.load() || required_size == nullptr) return -1;
    uint32_t size = 0;
    SDKReturnCode code = CoreSdk_GetGloveCalibrationSize(glove_id, &size);
    if (code != SDKReturnCode_Success) {
        SetError("CoreSdk_GetGloveCalibrationSize", code);
        return -1;
    }
    *required_size = size;
    if (output == nullptr) {
        g_last_error.clear();
        return 0;
    }
    if (capacity < size) {
        g_last_error = "Glove calibration output buffer is too small";
        return -1;
    }
    code = CoreSdk_GetGloveCalibration(output, size);
    if (code != SDKReturnCode_Success) {
        SetError("CoreSdk_GetGloveCalibration", code);
        return -1;
    }
    g_last_error.clear();
    return 0;
}

int manus_bridge_calibration_import(
    uint32_t glove_id, unsigned char* data, uint32_t size) {
    if (!g_connected.load() || data == nullptr || size == 0) return -1;
    SetGloveCalibrationReturnCode result = SetGloveCalibrationReturnCode_Error;
    const SDKReturnCode code = CoreSdk_SetGloveCalibration(
        glove_id, data, size, &result);
    if (code != SDKReturnCode_Success) {
        SetError("CoreSdk_SetGloveCalibration", code);
        return -1;
    }
    if (result != SetGloveCalibrationReturnCode_Success) {
        g_last_error = "CoreSdk_SetGloveCalibration rejected data (result=" +
                       std::to_string(static_cast<int>(result)) + ")";
        return -1;
    }
    g_last_error.clear();
    return 0;
}

int manus_bridge_poll(ManusBridgeFrame* output) {
    if (output == nullptr || !g_connected.load()) return 0;
    std::vector<RawFrame> frames;
    std::vector<ErgonomicsFrame> ergonomics;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (g_latest_frames.empty()) return 0;
        frames = g_latest_frames;
        ergonomics = g_latest_ergonomics;
    }

    // Query semantic topology outside the SDK callback, matching the official
    // examples. Prefer Left because this project's retarget path drives a left
    // MMHand; still return the first valid glove when no left glove is present.
    std::size_t selected = 0;
    std::vector<std::vector<NodeInfo>> all_info(frames.size());
    for (std::size_t index = 0; index < frames.size(); ++index) {
        RawFrame& candidate = frames[index];
        if (candidate.nodes.size() > kMaxNodes) return -1;
        uint32_t info_count = 0;
        SDKReturnCode code = CoreSdk_GetRawSkeletonNodeCount(
            candidate.glove_id, info_count);
        if (code != SDKReturnCode_Success || info_count != candidate.nodes.size()) {
            SetError("CoreSdk_GetRawSkeletonNodeCount", code);
            return -1;
        }
        all_info[index].resize(info_count);
        code = CoreSdk_GetRawSkeletonNodeInfoArray(
            candidate.glove_id, all_info[index].data(), info_count);
        if (code != SDKReturnCode_Success) {
            SetError("CoreSdk_GetRawSkeletonNodeInfoArray", code);
            return -1;
        }
        if (!all_info[index].empty() &&
            all_info[index].front().side == Side_Left) selected = index;
    }
    RawFrame& frame = frames[selected];
    std::vector<NodeInfo>& info = all_info[selected];

    std::memset(output, 0, sizeof(*output));
    output->sequence = frame.sequence;
    output->publish_time = frame.publish_time;
    output->glove_id = frame.glove_id;
    output->node_count = static_cast<uint32_t>(frame.nodes.size());
    output->side = Side_Invalid;
    for (std::size_t row = 0; row < frame.nodes.size(); ++row) {
        const SkeletonNode& source = frame.nodes[row];
        const auto match = std::find_if(
            info.begin(), info.end(), [&](const NodeInfo& item) {
                return item.nodeId == source.id;
            });
        if (match == info.end()) {
            g_last_error = "Raw Skeleton node ID is absent from NodeInfo";
            return -1;
        }
        ManusBridgeNode& target = output->nodes[row];
        target.node_id = source.id;
        target.parent_id = match->parentId;
        target.side = static_cast<int32_t>(match->side);
        target.chain_type = static_cast<int32_t>(match->chainType);
        target.finger_joint_type = static_cast<int32_t>(match->fingerJointType);
        target.position[0] = source.transform.position.x;
        target.position[1] = source.transform.position.y;
        target.position[2] = source.transform.position.z;
        target.rotation_wxyz[0] = source.transform.rotation.w;
        target.rotation_wxyz[1] = source.transform.rotation.x;
        target.rotation_wxyz[2] = source.transform.rotation.y;
        target.rotation_wxyz[3] = source.transform.rotation.z;
        if (output->side == Side_Invalid) output->side = target.side;
    }
    const auto ergonomics_match = std::find_if(
        ergonomics.begin(), ergonomics.end(), [&](const ErgonomicsFrame& item) {
            return !item.is_user_id && item.id == frame.glove_id;
        });
    if (ergonomics_match != ergonomics.end() &&
        (output->side == Side_Left || output->side == Side_Right)) {
        const std::size_t offset = output->side == Side_Left ? 0 : kHandErgonomicsCount;
        std::copy_n(
            ergonomics_match->data.begin() + offset,
            kHandErgonomicsCount,
            output->ergonomics);
        output->has_ergonomics = 1;
    }
    return 1;
}

const char* manus_bridge_last_error() { return g_last_error.c_str(); }

void manus_bridge_shutdown() {
    if (!g_initialized) return;
    CoreSdk_ShutDown();
    g_initialized = false;
    g_connected.store(false);
    std::lock_guard<std::mutex> lock(g_mutex);
    g_latest_frames.clear();
    g_latest_ergonomics.clear();
    g_sequence = 0;
}
}
