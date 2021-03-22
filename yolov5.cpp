#include <iostream>
#include <chrono>
#include "cuda_runtime_api.h"
#include "logging.h"
#include "common.hpp"
#include"flycapture/FlyCapture2.h"
#include <thread>
#include <unistd.h>
#include <stdio.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <arpa/inet.h>
#include "opencv2/video/tracking.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/tracking.hpp"
#include <X11/Xlib.h>

#define IP_FOUND  "IP_FOUND"
#define IP_FOUND_ACK  "IP_FOUND_ACK"
#define MCAST "224.0.0.88"
#define MCAST1 "224.0.0.87"


#define USE_FP16  // comment out this if want to use FP32
#define DEVICE 0  // GPU id
#define DEVICE1 1
#define NMS_THRESH 0.5
#define CONF_THRESH 0.6
#define BATCH_SIZE 1

#define NET x  // s m l x
#define NETSTRUCT(str) createEngine_##str
#define CREATENET(net) NETSTRUCT(net)
#define STR1(x) #x
#define STR2(x) STR1(x)

// stuff we know about the network and the input/output blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int CLASS_NUM = Yolo::CLASS_NUM;
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;

// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine_s(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("../yolov5s.wts");
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    // yolov5 backbone
    auto focus0 = focus(network, weightMap, *data, 3, 32, 3, "model.0");
    auto conv1 = convBlock(network, weightMap, *focus0->getOutput(0), 64, 3, 2, 1, "model.1");
    auto bottleneck_CSP2 = bottleneckCSP(network, weightMap, *conv1->getOutput(0), 64, 64, 1, true, 1, 0.5, "model.2");
    auto conv3 = convBlock(network, weightMap, *bottleneck_CSP2->getOutput(0), 128, 3, 2, 1, "model.3");
    auto bottleneck_csp4 = bottleneckCSP(network, weightMap, *conv3->getOutput(0), 128, 128, 3, true, 1, 0.5, "model.4");
    auto conv5 = convBlock(network, weightMap, *bottleneck_csp4->getOutput(0), 256, 3, 2, 1, "model.5");
    auto bottleneck_csp6 = bottleneckCSP(network, weightMap, *conv5->getOutput(0), 256, 256, 3, true, 1, 0.5, "model.6");
    auto conv7 = convBlock(network, weightMap, *bottleneck_csp6->getOutput(0), 512, 3, 2, 1, "model.7");
    auto spp8 = SPP(network, weightMap, *conv7->getOutput(0), 512, 512, 5, 9, 13, "model.8");

    // yolov5 head
    auto bottleneck_csp9 = bottleneckCSP(network, weightMap, *spp8->getOutput(0), 512, 512, 1, false, 1, 0.5, "model.9");
    auto conv10 = convBlock(network, weightMap, *bottleneck_csp9->getOutput(0), 256, 1, 1, 1, "model.10");

    float *deval = reinterpret_cast<float*>(malloc(sizeof(float) * 256 * 2 * 2));
    for (int i = 0; i < 256 * 2 * 2; i++) {
        deval[i] = 1.0;
    }
    Weights deconvwts11{ DataType::kFLOAT, deval, 256 * 2 * 2 };
    IDeconvolutionLayer* deconv11 = network->addDeconvolutionNd(*conv10->getOutput(0), 256, DimsHW{ 2, 2 }, deconvwts11, emptywts);
    deconv11->setStrideNd(DimsHW{ 2, 2 });
    deconv11->setNbGroups(256);
    weightMap["deconv11"] = deconvwts11;

    ITensor* inputTensors12[] = { deconv11->getOutput(0), bottleneck_csp6->getOutput(0) };
    auto cat12 = network->addConcatenation(inputTensors12, 2);
    auto bottleneck_csp13 = bottleneckCSP(network, weightMap, *cat12->getOutput(0), 512, 256, 1, false, 1, 0.5, "model.13");
    auto conv14 = convBlock(network, weightMap, *bottleneck_csp13->getOutput(0), 128, 1, 1, 1, "model.14");

    Weights deconvwts15{ DataType::kFLOAT, deval, 128 * 2 * 2 };
    IDeconvolutionLayer* deconv15 = network->addDeconvolutionNd(*conv14->getOutput(0), 128, DimsHW{ 2, 2 }, deconvwts15, emptywts);
    deconv15->setStrideNd(DimsHW{ 2, 2 });
    deconv15->setNbGroups(128);

    ITensor* inputTensors16[] = { deconv15->getOutput(0), bottleneck_csp4->getOutput(0) };
    auto cat16 = network->addConcatenation(inputTensors16, 2);
    auto bottleneck_csp17 = bottleneckCSP(network, weightMap, *cat16->getOutput(0), 256, 128, 1, false, 1, 0.5, "model.17");
    IConvolutionLayer* det0 = network->addConvolutionNd(*bottleneck_csp17->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.0.weight"], weightMap["model.24.m.0.bias"]);

    auto conv18 = convBlock(network, weightMap, *bottleneck_csp17->getOutput(0), 128, 3, 2, 1, "model.18");
    ITensor* inputTensors19[] = { conv18->getOutput(0), conv14->getOutput(0) };
    auto cat19 = network->addConcatenation(inputTensors19, 2);
    auto bottleneck_csp20 = bottleneckCSP(network, weightMap, *cat19->getOutput(0), 256, 256, 1, false, 1, 0.5, "model.20");
    IConvolutionLayer* det1 = network->addConvolutionNd(*bottleneck_csp20->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.1.weight"], weightMap["model.24.m.1.bias"]);

    auto conv21 = convBlock(network, weightMap, *bottleneck_csp20->getOutput(0), 256, 3, 2, 1, "model.21");
    ITensor* inputTensors22[] = { conv21->getOutput(0), conv10->getOutput(0) };
    auto cat22 = network->addConcatenation(inputTensors22, 2);
    auto bottleneck_csp23 = bottleneckCSP(network, weightMap, *cat22->getOutput(0), 512, 512, 1, false, 1, 0.5, "model.23");
    IConvolutionLayer* det2 = network->addConvolutionNd(*bottleneck_csp23->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.2.weight"], weightMap["model.24.m.2.bias"]);

    auto yolo = addYoLoLayer(network, weightMap, det0, det1, det2);
    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*yolo->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#ifdef USE_FP16
    config->setFlag(BuilderFlag::kFP16);
#endif
    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.values));

    }
    return engine;
}

ICudaEngine* createEngine_m(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("../yolov5m.wts");
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    /* ------ yolov5 backbone------ */
    auto focus0 = focus(network, weightMap, *data, 3, 48, 3, "model.0");
    auto conv1 = convBlock(network, weightMap, *focus0->getOutput(0), 96, 3, 2, 1, "model.1");
    auto bottleneck_CSP2 = bottleneckCSP(network, weightMap, *conv1->getOutput(0), 96, 96, 2, true, 1, 0.5, "model.2");
    auto conv3 = convBlock(network, weightMap, *bottleneck_CSP2->getOutput(0), 192, 3, 2, 1, "model.3");
    auto bottleneck_csp4 = bottleneckCSP(network, weightMap, *conv3->getOutput(0), 192, 192, 6, true, 1, 0.5, "model.4");
    auto conv5 = convBlock(network, weightMap, *bottleneck_csp4->getOutput(0), 384, 3, 2, 1, "model.5");
    auto bottleneck_csp6 = bottleneckCSP(network, weightMap, *conv5->getOutput(0), 384, 384, 6, true, 1, 0.5, "model.6");
    auto conv7 = convBlock(network, weightMap, *bottleneck_csp6->getOutput(0), 768, 3, 2, 1, "model.7");
    auto spp8 = SPP(network, weightMap, *conv7->getOutput(0), 768, 768, 5, 9, 13, "model.8");
    /* ------ yolov5 head ------ */
    auto bottleneck_csp9 = bottleneckCSP(network, weightMap, *spp8->getOutput(0), 768, 768, 2, false, 1, 0.5, "model.9");
    auto conv10 = convBlock(network, weightMap, *bottleneck_csp9->getOutput(0), 384, 1, 1, 1, "model.10");

    float *deval = reinterpret_cast<float*>(malloc(sizeof(float) * 384 * 2 * 2));
    for (int i = 0; i < 384 * 2 * 2; i++) {
        deval[i] = 1.0;
    }
    Weights deconvwts11{ DataType::kFLOAT, deval, 384 * 2 * 2 };
    IDeconvolutionLayer* deconv11 = network->addDeconvolutionNd(*conv10->getOutput(0), 384, DimsHW{ 2, 2 }, deconvwts11, emptywts);
    deconv11->setStrideNd(DimsHW{ 2, 2 });
    deconv11->setNbGroups(384);
    weightMap["deconv11"] = deconvwts11;
    ITensor* inputTensors12[] = { deconv11->getOutput(0), bottleneck_csp6->getOutput(0) };
    auto cat12 = network->addConcatenation(inputTensors12, 2);

    auto bottleneck_csp13 = bottleneckCSP(network, weightMap, *cat12->getOutput(0), 768, 384, 2, false, 1, 0.5, "model.13");

    auto conv14 = convBlock(network, weightMap, *bottleneck_csp13->getOutput(0), 192, 1, 1, 1, "model.14");

    Weights deconvwts15{ DataType::kFLOAT, deval, 192 * 2 * 2 };
    IDeconvolutionLayer* deconv15 = network->addDeconvolutionNd(*conv14->getOutput(0), 192, DimsHW{ 2, 2 }, deconvwts15, emptywts);
    deconv15->setStrideNd(DimsHW{ 2, 2 });
    deconv15->setNbGroups(192);

    ITensor* inputTensors16[] = { deconv15->getOutput(0), bottleneck_csp4->getOutput(0) };
    auto cat16 = network->addConcatenation(inputTensors16, 2);
    auto bottleneck_csp17 = bottleneckCSP(network, weightMap, *cat16->getOutput(0), 384, 192, 2, false, 1, 0.5, "model.17");

    //yolo layer 0
    IConvolutionLayer* det0 = network->addConvolutionNd(*bottleneck_csp17->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.0.weight"], weightMap["model.24.m.0.bias"]);
    auto conv18 = convBlock(network, weightMap, *bottleneck_csp17->getOutput(0), 192, 3, 2, 1, "model.18");
    ITensor* inputTensors19[] = { conv18->getOutput(0), conv14->getOutput(0) };
    auto cat19 = network->addConcatenation(inputTensors19, 2);
    auto bottleneck_csp20 = bottleneckCSP(network, weightMap, *cat19->getOutput(0), 384, 384, 2, false, 1, 0.5, "model.20");

    //yolo layer 1
    IConvolutionLayer* det1 = network->addConvolutionNd(*bottleneck_csp20->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.1.weight"], weightMap["model.24.m.1.bias"]);
    auto conv21 = convBlock(network, weightMap, *bottleneck_csp20->getOutput(0), 384, 3, 2, 1, "model.21");
    ITensor* inputTensors22[] = { conv21->getOutput(0), conv10->getOutput(0) };
    auto cat22 = network->addConcatenation(inputTensors22, 2);
    auto bottleneck_csp23 = bottleneckCSP(network, weightMap, *cat22->getOutput(0), 768, 768, 2, false, 1, 0.5, "model.23");
    // yolo layer 2
    IConvolutionLayer* det2 = network->addConvolutionNd(*bottleneck_csp23->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.2.weight"], weightMap["model.24.m.2.bias"]);

    auto yolo = addYoLoLayer(network, weightMap, det0, det1, det2);
    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*yolo->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#ifdef USE_FP16
    config->setFlag(BuilderFlag::kFP16);
#endif
    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.values));
    }

    return engine;
}

ICudaEngine* createEngine_l(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("../yolov5l.wts");
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    /* ------ yolov5 backbone------ */
    auto focus0 = focus(network, weightMap, *data, 3, 64, 3, "model.0");
    auto conv1 = convBlock(network, weightMap, *focus0->getOutput(0), 128, 3, 2, 1, "model.1");
    auto bottleneck_CSP2 = bottleneckCSP(network, weightMap, *conv1->getOutput(0), 128, 128, 3, true, 1, 0.5, "model.2");
    auto conv3 = convBlock(network, weightMap, *bottleneck_CSP2->getOutput(0), 256, 3, 2, 1, "model.3");
    auto bottleneck_csp4 = bottleneckCSP(network, weightMap, *conv3->getOutput(0), 256, 256, 9, true, 1, 0.5, "model.4");
    auto conv5 = convBlock(network, weightMap, *bottleneck_csp4->getOutput(0), 512, 3, 2, 1, "model.5");
    auto bottleneck_csp6 = bottleneckCSP(network, weightMap, *conv5->getOutput(0), 512, 512, 9, true, 1, 0.5, "model.6");
    auto conv7 = convBlock(network, weightMap, *bottleneck_csp6->getOutput(0), 1024, 3, 2, 1, "model.7");
    auto spp8 = SPP(network, weightMap, *conv7->getOutput(0), 1024, 1024, 5, 9, 13, "model.8");

    /* ------ yolov5 head ------ */
    auto bottleneck_csp9 = bottleneckCSP(network, weightMap, *spp8->getOutput(0), 1024, 1024, 3, false, 1, 0.5, "model.9");
    auto conv10 = convBlock(network, weightMap, *bottleneck_csp9->getOutput(0), 512, 1, 1, 1, "model.10");

    float *deval = reinterpret_cast<float*>(malloc(sizeof(float) * 512 * 2 * 2));
    for (int i = 0; i < 512 * 2 * 2; i++) {
        deval[i] = 1.0;
    }
    Weights deconvwts11{ DataType::kFLOAT, deval, 512 * 2 * 2 };
    IDeconvolutionLayer* deconv11 = network->addDeconvolutionNd(*conv10->getOutput(0), 512, DimsHW{ 2, 2 }, deconvwts11, emptywts);
    deconv11->setStrideNd(DimsHW{ 2, 2 });
    deconv11->setNbGroups(512);
    weightMap["deconv11"] = deconvwts11;

    ITensor* inputTensors12[] = { deconv11->getOutput(0), bottleneck_csp6->getOutput(0) };
    auto cat12 = network->addConcatenation(inputTensors12, 2);
    auto bottleneck_csp13 = bottleneckCSP(network, weightMap, *cat12->getOutput(0), 1024, 512, 3, false, 1, 0.5, "model.13");
    auto conv14 = convBlock(network, weightMap, *bottleneck_csp13->getOutput(0), 256, 1, 1, 1, "model.14");

    Weights deconvwts15{ DataType::kFLOAT, deval, 256 * 2 * 2 };
    IDeconvolutionLayer* deconv15 = network->addDeconvolutionNd(*conv14->getOutput(0), 256, DimsHW{ 2, 2 }, deconvwts15, emptywts);
    deconv15->setStrideNd(DimsHW{ 2, 2 });
    deconv15->setNbGroups(256);
    ITensor* inputTensors16[] = { deconv15->getOutput(0), bottleneck_csp4->getOutput(0) };
    auto cat16 = network->addConcatenation(inputTensors16, 2);

    auto bottleneck_csp17 = bottleneckCSP(network, weightMap, *cat16->getOutput(0), 512, 256, 3, false, 1, 0.5, "model.17");

    // yolo layer 0
    IConvolutionLayer* det0 = network->addConvolutionNd(*bottleneck_csp17->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.0.weight"], weightMap["model.24.m.0.bias"]);
    auto conv18 = convBlock(network, weightMap, *bottleneck_csp17->getOutput(0), 256, 3, 2, 1, "model.18");
    ITensor* inputTensors19[] = { conv18->getOutput(0), conv14->getOutput(0) };
    auto cat19 = network->addConcatenation(inputTensors19, 2);
    auto bottleneck_csp20 = bottleneckCSP(network, weightMap, *cat19->getOutput(0), 512, 512, 3, false, 1, 0.5, "model.20");
    //yolo layer 1
    IConvolutionLayer* det1 = network->addConvolutionNd(*bottleneck_csp20->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.1.weight"], weightMap["model.24.m.1.bias"]);
    auto conv21 = convBlock(network, weightMap, *bottleneck_csp20->getOutput(0), 512, 3, 2, 1, "model.21");
    ITensor* inputTensors22[] = { conv21->getOutput(0), conv10->getOutput(0) };
    auto cat22 = network->addConcatenation(inputTensors22, 2);
    auto bottleneck_csp23 = bottleneckCSP(network, weightMap, *cat22->getOutput(0), 1024, 1024, 3, false, 1, 0.5, "model.23");
    IConvolutionLayer* det2 = network->addConvolutionNd(*bottleneck_csp23->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.2.weight"], weightMap["model.24.m.2.bias"]);

    auto yolo = addYoLoLayer(network, weightMap, det0, det1, det2);
    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*yolo->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#ifdef USE_FP16
    config->setFlag(BuilderFlag::kFP16);
#endif
    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.values));
    }

    return engine;
}

ICudaEngine* createEngine_x(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("../yolov5x.wts");
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    /* ------ yolov5 backbone------ */
    auto focus0 = focus(network, weightMap, *data, 3, 80, 3, "model.0");
    auto conv1 = convBlock(network, weightMap, *focus0->getOutput(0), 160, 3, 2, 1, "model.1");
    auto bottleneck_CSP2 = bottleneckCSP(network, weightMap, *conv1->getOutput(0), 160, 160, 4, true, 1, 0.5, "model.2");
    auto conv3 = convBlock(network, weightMap, *bottleneck_CSP2->getOutput(0), 320, 3, 2, 1, "model.3");
    auto bottleneck_csp4 = bottleneckCSP(network, weightMap, *conv3->getOutput(0), 320, 320, 12, true, 1, 0.5, "model.4");
    auto conv5 = convBlock(network, weightMap, *bottleneck_csp4->getOutput(0), 640, 3, 2, 1, "model.5");
    auto bottleneck_csp6 = bottleneckCSP(network, weightMap, *conv5->getOutput(0), 640, 640, 12, true, 1, 0.5, "model.6");
    auto conv7 = convBlock(network, weightMap, *bottleneck_csp6->getOutput(0), 1280, 3, 2, 1, "model.7");
    auto spp8 = SPP(network, weightMap, *conv7->getOutput(0), 1280, 1280, 5, 9, 13, "model.8");

    /* ------- yolov5 head ------- */
    auto bottleneck_csp9 = bottleneckCSP(network, weightMap, *spp8->getOutput(0), 1280, 1280, 4, false, 1, 0.5, "model.9");
    auto conv10 = convBlock(network, weightMap, *bottleneck_csp9->getOutput(0), 640, 1, 1, 1, "model.10");

    float *deval = reinterpret_cast<float*>(malloc(sizeof(float) * 640 * 2 * 2));
    for (int i = 0; i < 640 * 2 * 2; i++) {
        deval[i] = 1.0;
    }
    Weights deconvwts11{ DataType::kFLOAT, deval, 640 * 2 * 2 };
    IDeconvolutionLayer* deconv11 = network->addDeconvolutionNd(*conv10->getOutput(0), 640, DimsHW{ 2, 2 }, deconvwts11, emptywts);
    deconv11->setStrideNd(DimsHW{ 2, 2 });
    deconv11->setNbGroups(640);
    weightMap["deconv11"] = deconvwts11;

    ITensor* inputTensors12[] = { deconv11->getOutput(0), bottleneck_csp6->getOutput(0) };
    auto cat12 = network->addConcatenation(inputTensors12, 2);

    auto bottleneck_csp13 = bottleneckCSP(network, weightMap, *cat12->getOutput(0), 1280, 640, 4, false, 1, 0.5, "model.13");
    auto conv14 = convBlock(network, weightMap, *bottleneck_csp13->getOutput(0), 320, 1, 1, 1, "model.14");

    Weights deconvwts15{ DataType::kFLOAT, deval, 320 * 2 * 2 };
    IDeconvolutionLayer* deconv15 = network->addDeconvolutionNd(*conv14->getOutput(0), 320, DimsHW{ 2, 2 }, deconvwts15, emptywts);
    deconv15->setStrideNd(DimsHW{ 2, 2 });
    deconv15->setNbGroups(320);
    ITensor* inputTensors16[] = { deconv15->getOutput(0), bottleneck_csp4->getOutput(0) };
    auto cat16 = network->addConcatenation(inputTensors16, 2);

    auto bottleneck_csp17 = bottleneckCSP(network, weightMap, *cat16->getOutput(0), 640, 320, 4, false, 1, 0.5, "model.17");

    // yolo layer 0
    IConvolutionLayer* det0 = network->addConvolutionNd(*bottleneck_csp17->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.0.weight"], weightMap["model.24.m.0.bias"]);
    auto conv18 = convBlock(network, weightMap, *bottleneck_csp17->getOutput(0), 320, 3, 2, 1, "model.18");
    ITensor* inputTensors19[] = { conv18->getOutput(0), conv14->getOutput(0) };
    auto cat19 = network->addConcatenation(inputTensors19, 2);
    auto bottleneck_csp20 = bottleneckCSP(network, weightMap, *cat19->getOutput(0), 640, 640, 4, false, 1, 0.5, "model.20");
    // yolo layer 1
    IConvolutionLayer* det1 = network->addConvolutionNd(*bottleneck_csp20->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.1.weight"], weightMap["model.24.m.1.bias"]);
    auto conv21 = convBlock(network, weightMap, *bottleneck_csp20->getOutput(0), 640, 3, 2, 1, "model.21");
    ITensor* inputTensors22[] = { conv21->getOutput(0), conv10->getOutput(0) };
    auto cat22 = network->addConcatenation(inputTensors22, 2);
    auto bottleneck_csp23 = bottleneckCSP(network, weightMap, *cat22->getOutput(0), 1280, 1280, 4, false, 1, 0.5, "model.23");
    // yolo layer 2
    IConvolutionLayer* det2 = network->addConvolutionNd(*bottleneck_csp23->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.2.weight"], weightMap["model.24.m.2.bias"]);

    auto yolo = addYoLoLayer(network, weightMap, det0, det1, det2);
    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*yolo->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#ifdef USE_FP16
    config->setFlag(BuilderFlag::kFP16);
#endif
    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.values));
    }

    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = (CREATENET(NET))(maxBatchSize, builder, config, DataType::kFLOAT);
    //ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
}

void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* output, int batchSize) {
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}



using namespace cv;
using namespace FlyCapture2;
using namespace std;
Camera* pCameras;
Mat mask(3, 3, CV_64FC1);
Mat org(3, 1, CV_64FC1);
Mat res(3, 1, CV_64FC1);

Mat org1(3, 1, CV_64FC1);
Mat res1(3, 1, CV_64FC1);

Mat mask2(3, 3, CV_64FC1);
Mat org2(3, 1, CV_64FC1);
Mat res2(3, 1, CV_64FC1);

Mat org3(3, 1, CV_64FC1);
Mat res3(3, 1, CV_64FC1);
struct SgData 
{
    bool car1;//blue1
    bool car2;//blue2
    bool car3;//red1
    bool car4;///red2
    float car1_x;
    float car1_y;
    float car2_x;
    float car2_y;
    float car3_x;
    float car3_y;
    float car4_x;
    float car4_y;
}sgdata;

int k = 0;
static Point edge;
Point a, b, c, d;

double post_1_blue_1x =0 ;
double post_1_blue_2x =0 ;
double post_1_blue_1y =0 ;
double post_1_blue_2y =0 ;
double post_2_blue_1y =0 ;
double post_2_blue_1x =0 ;
double post_2_blue_2x =0 ;
double post_2_blue_2y =0 ;

double post_1_red_1x =0 ;
double post_1_red_2x =0 ;
double post_1_red_1y =0 ;
double post_1_red_2y =0 ;
double post_2_red_1y =0 ;
double post_2_red_1x =0 ;
double post_2_red_2x =0 ;
double post_2_red_2y =0 ;

int enemy_color=1;

void PrintError(FlyCapture2::Error error) { error.PrintErrorTrace(); }
int cam_initialize() {

    //PrintBuildInfo();

    //const int k_numImages = 50;
    FlyCapture2::Error error;

    BusManager busMgr;
    printf("1");
    unsigned int numCameras;
    error = busMgr.GetNumOfCameras(&numCameras);
    if (error != PGRERROR_OK)
    {
        PrintError(error);
        return -1;
    }

    std::cout << "Number of cameras detected: " << numCameras << endl;
    /*
    if (numCameras < 2)
    {
        std::cout << "Insufficient number of cameras." << endl;
        std::cout << "Make sure at least two cameras are connected for example to "
            "run."
            << endl;
        std::cout << "Press Enter to exit." << endl;
        cin.ignore();
        return -1;
    }
    */
    pCameras = new Camera[numCameras];


    for (unsigned int i = 0; i < numCameras; i++)
    {
        PGRGuid guid;
        error = busMgr.GetCameraFromIndex(i, &guid);
        if (error != PGRERROR_OK)
        {
            PrintError(error);
            std::cout << "Press Enter to exit." << endl;
            delete[] pCameras;
            cin.ignore();
            return -1;
        }

        // Connect to a camera
        error = pCameras[i].Connect(&guid);
        if (error != PGRERROR_OK)
        {
            PrintError(error);
            delete[] pCameras;
            std::cout << "Press Enter to exit." << endl;
            cin.ignore();
            return -1;
        }

        // Get the camera information
        CameraInfo camInfo;
        error = pCameras[i].GetCameraInfo(&camInfo);
        if (error != PGRERROR_OK)
        {
            PrintError(error);
            delete[] pCameras;
            std::cout << "Press Enter to exit." << endl;
            cin.ignore();
            return -1;
        }

        //PrintCameraInfo(&camInfo);

        // Turn trigger mode off
        TriggerMode trigMode;
        trigMode.onOff = false;
        error = pCameras[i].SetTriggerMode(&trigMode);
        if (error != PGRERROR_OK)
        {
            PrintError(error);
            delete[] pCameras;
            std::cout << "Press Enter to exit." << endl;
            cin.ignore();
            return -1;
        }

        // Turn Timestamp on
        EmbeddedImageInfo imageInfo;
        imageInfo.timestamp.onOff = true;
        error = pCameras[i].SetEmbeddedImageInfo(&imageInfo);
        if (error != PGRERROR_OK)
        {
            PrintError(error);
            delete[] pCameras;
            std::cout << "Press Enter to exit." << endl;
            cin.ignore();
            return -1;
        }

        // Start streaming on camera
        error = pCameras[i].StartCapture();
        if (error != PGRERROR_OK)
        {
            PrintError(error);
            delete[] pCameras;
            std::cout << "Press Enter to exit." << endl;
            cin.ignore();
            return -1;
        }
    }
}

static void onMouse1(int event, int x, int y, int, void* userInput)
{
    //printf("mouse1");
    Mat src, dst;
    dst.copyTo(src);
    if (event != EVENT_LBUTTONDOWN)
        return;
    // Get the pointer input image
    Mat* img = (Mat*)userInput;
    // Draw circle
    circle(*img, Point(x, y), 5, Scalar(0, 255, 0), 3);

    src.copyTo(dst);
    //edge = Point(x * 1.25, y * 1.25);
    edge=Point(x,y);
    if (k >= 0 && k <= 3) {
        std::cout << "x:" << x << "y:" << y << endl;
    }

    src.copyTo(dst);//确保画线操作是在src上进行

    k = k + 1;
    if (k > 0) {
        if (k == 1) {
            a.x = edge.x;
            a.y = edge.y;
        }
        if (k == 2) {
            b.x = edge.x;
            b.y = edge.y;
        }
        if (k == 3) {
            c.x = edge.x;
            c.y = edge.y;
        }
        if (k == 4) {
            d.x = edge.x;
            d.y = edge.y;
        }
        if (k == 5) {
            //CvMat* mask = cvCreateMat(3, 3, CV_32FC1);
            Point2f camera_view[] = { a,b,c,d };
            Point2f god_view[] = { Point2f(808,448),Point2f(808,0),Point2f(0,448),Point2f(0,0) };
            //计算变换矩阵
            mask = getPerspectiveTransform(camera_view, god_view);
            std::cout << mask << endl;
        }
    }
}
static void onMouse2(int event, int x, int y, int, void* userInput)
{
    //printf("onmouse2");
    Mat src, dst;
    dst.copyTo(src);
    if (event != EVENT_LBUTTONDOWN)
        return;
    // Get the pointer input image
    Mat* img = (Mat*)userInput;
    // Draw circle
    circle(*img, Point(x, y), 5, Scalar(0, 255, 0), 3);
    edge = Point(x , y );
    if (k >= 5 && k <= 8) {
        std::cout << "x:" << x << "y:" << y << endl;
    }
    //putText(src, temp_1, Point(x, y), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(150, 55, 245));
    //imshow("src", src);
    src.copyTo(dst);//确保画线操作是在src上进行

    k = k + 1;
    if (k > 0) {
        if (k == 6) {
            a.x = edge.x;
            a.y = edge.y;
        }
        if (k == 7) {
            b.x = edge.x;
            b.y = edge.y;
        }
        if (k == 8) {
            c.x = edge.x;
            c.y = edge.y;
        }
        if (k == 9) {
            d.x = edge.x;
            d.y = edge.y;
        }
        if (k == 10) {
            //CvMat* mask = cvCreateMat(3, 3, CV_32FC1);
            Point2f camera_view[] = { a,b,c,d };
            //Point2f god_view[] = { Point2f(0,0),Point2f(0,448),Point2f(808,0),Point2f(808,448) };
            Point2f god_view[] = { Point2f(808,448),Point2f(808,0),Point2f(0,448),Point2f(0,0) };
            mask2 = getPerspectiveTransform(camera_view, god_view);
            std::cout << mask2 << endl;
        }
    }

}

const int stateNum=4;
const int measureNum=2;
RNG rng;
KalmanFilter KF_1_1(stateNum,measureNum,0);
KalmanFilter KF_1_2(stateNum,measureNum,0);
KalmanFilter KF_2_1(stateNum,measureNum,0);
KalmanFilter KF_2_2(stateNum,measureNum,0);
KalmanFilter KF_1_3(stateNum,measureNum,0);
KalmanFilter KF_1_4(stateNum,measureNum,0);
KalmanFilter KF_2_3(stateNum,measureNum,0);
KalmanFilter KF_2_4(stateNum,measureNum,0);
Mat measurement = Mat::zeros(measureNum,1,CV_32F);
Mat measurement_1_2 = Mat::zeros(measureNum,1,CV_32F);
Mat measurement_2_1 = Mat::zeros(measureNum,1,CV_32F);
Mat measurement_2_2 = Mat::zeros(measureNum,1,CV_32F);
Mat measurement_2_3 = Mat::zeros(measureNum,1,CV_32F);
Mat measurement_2_4 = Mat::zeros(measureNum,1,CV_32F);
Mat measurement_1_3 = Mat::zeros(measureNum,1,CV_32F);
Mat measurement_1_4 = Mat::zeros(measureNum,1,CV_32F);
double kalman_1_1x=0,kalman_1_1y=0;
double kalman_1_2x=0,kalman_1_2y=0;
double kalman_2_1x=0,kalman_2_1y=0;
double kalman_2_2x=0,kalman_2_2y=0;
double kalman_2_3x=0,kalman_2_4y=0;
double kalman_2_4x=0,kalman_2_4y=0;
double kalman_1_3x=0,kalman_1_4y=0;
double kalman_1_4x=0,kalman_1_4y=0;

void kalman(double x,double y)
{
    Mat prediction=KF_1_1.predict();
    Point predict_pt = Point(prediction.at<float>(0), prediction.at<float>(1));
    measurement.at<float>(0) = (float)x;
	measurement.at<float>(1) = (float)y;
	KF_1_1.correct(measurement);

    cout<<"kalman predict car1: "<<prediction.at<float>(0)<<" "<<prediction.at<float>(1)<<endl;
    kalman_1_1x=prediction.at<float>(0);
    kalman_1_1y=prediction.at<float>(1);
}
void kalman_1_2(double x,double y)
{
    Mat prediction=KF_1_2.predict();
    Point predict_pt = Point(prediction.at<float>(0), prediction.at<float>(1));
    measurement_1_2.at<float>(0) = (float)x;
	measurement_1_2.at<float>(1) = (float)y;
	KF_1_2.correct(measurement_1_2);

    cout<<"kalman predict car2: "<<prediction.at<float>(0)<<" "<<prediction.at<float>(1)<<endl;
    kalman_1_2x=prediction.at<float>(0);
    kalman_1_2y=prediction.at<float>(1);
}
void kalman_2_1(double x,double y)
{
    Mat prediction=KF_2_1.predict();
    Point predict_pt = Point(prediction.at<float>(0), prediction.at<float>(1));
    measurement_2_1.at<float>(0) = (float)x;
	measurement_2_1.at<float>(1) = (float)y;
	KF_2_1.correct(measurement_2_1);

    cout<<"kalman predict car1: "<<prediction.at<float>(0)<<" "<<prediction.at<float>(1)<<endl;
    kalman_2_1x=prediction.at<float>(0);
    kalman_2_1y=prediction.at<float>(1);
}
void kalman_2_2(double x,double y)
{
    Mat prediction=KF_2_2.predict();
    Point predict_pt = Point(prediction.at<float>(0), prediction.at<float>(1));
    measurement_2_2.at<float>(0) = (float)x;
	measurement_2_2.at<float>(1) = (float)y;
	KF_2_2.correct(measurement_2_2);

    cout<<"kalman predict car2: "<<prediction.at<float>(0)<<" "<<prediction.at<float>(1)<<endl;
    kalman_2_2x=prediction.at<float>(0);
    kalman_2_2y=prediction.at<float>(1);
}
void kalman_1_3(double x,double y)
{
    Mat prediction=KF_1_3.predict();
    Point predict_pt = Point(prediction.at<float>(0), prediction.at<float>(1));
    measurement_1_3.at<float>(0) = (float)x;
	measurement_1_3.at<float>(1) = (float)y;
	KF_1_3.correct(measurement_1_3);

    cout<<"kalman predict car2: "<<prediction.at<float>(0)<<" "<<prediction.at<float>(1)<<endl;
    kalman_1_3x=prediction.at<float>(0);
    kalman_1_3y=prediction.at<float>(1);
}
void kalman_1_4(double x,double y)
{
    Mat prediction=KF_1_4.predict();
    Point predict_pt = Point(prediction.at<float>(0), prediction.at<float>(1));
    measurement_1_4.at<float>(0) = (float)x;
	measurement_1_4.at<float>(1) = (float)y;
	KF_1_4.correct(measurement_1_4);

    cout<<"kalman predict car2: "<<prediction.at<float>(0)<<" "<<prediction.at<float>(1)<<endl;
    kalman_1_4x=prediction.at<float>(0);
    kalman_1_4y=prediction.at<float>(1);
}
void kalman_2_3(double x,double y)
{
    Mat prediction=KF_2_3.predict();
    Point predict_pt = Point(prediction.at<float>(0), prediction.at<float>(1));
    measurement_2_3.at<float>(0) = (float)x;
	measurement_2_3.at<float>(1) = (float)y;
	KF_2_3.correct(measurement_1_2);

    cout<<"kalman predict car2: "<<prediction.at<float>(0)<<" "<<prediction.at<float>(1)<<endl;
    kalman_2_3x=prediction.at<float>(0);
    kalman_2_3y=prediction.at<float>(1);
}
void kalman_2_4(double x,double y)
{
    Mat prediction=KF_2_4.predict();
    Point predict_pt = Point(prediction.at<float>(0), prediction.at<float>(1));
    measurement_2_4.at<float>(0) = (float)x;
	measurement_2_4.at<float>(1) = (float)y;
	KF_2_4.correct(measurement_2_4);

    cout<<"kalman predict car2: "<<prediction.at<float>(0)<<" "<<prediction.at<float>(1)<<endl;
    kalman_2_4x=prediction.at<float>(0);
    kalman_2_4y=prediction.at<float>(1);
}

double calculate_distance(double x,double y,double x1,double y1)
{
    return sqrt((x-x1)*(x-x1)+(y-y1)*(y-y1));
}
void infergai(int pattern, const char* img_path, std::string engine_name)
{
	printf("thread1");
    cudaSetDevice(DEVICE);
	// create a model using the API directly and serialize it to a stream
	char *trtModelStream{ nullptr };
	size_t size{ 0 };

	std::ifstream file(engine_name, std::ios::binary);
	if (file.good()) {
		file.seekg(0, file.end);
		size = file.tellg();
		file.seekg(0, file.beg);
		trtModelStream = new char[size];
		assert(trtModelStream);
		file.read(trtModelStream, size);
		file.close();
	}
	if (pattern == 1){
		std::vector<std::string> file_names;
		if (read_files_in_dir(img_path, file_names) < 0) {
			std::cout << "read_files_in_dir failed." << std::endl;
			return;
		}
		// prepare input data ---------------------------
		static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
		//for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
		//    data[i] = 1.0;
		static float prob[BATCH_SIZE * OUTPUT_SIZE];
		IRuntime* runtime = createInferRuntime(gLogger);
		assert(runtime != nullptr);
		ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
		assert(engine != nullptr);
		IExecutionContext* context = engine->createExecutionContext();
		assert(context != nullptr);
		delete[] trtModelStream;
		assert(engine->getNbBindings() == 2);
		void* buffers[2];
		// In order to bind the buffers, we need to know the names of the input and output tensors.
		// Note that indices are guaranteed to be less than IEngine::getNbBindings()
		const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
		const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
		assert(inputIndex == 0);
		assert(outputIndex == 1);
		// Create GPU buffers on device
		CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
		CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
		// Create stream
		cudaStream_t stream;
		CHECK(cudaStreamCreate(&stream));
		int fcount = 0;
		for (int f = 0; f < (int)file_names.size(); f++) { //fΪ\CEļ\FE\BC\D0\D6\D0ͼƬ\B5\C4index,Ϊfcount\B5ı\B6\CA\FD
			fcount++;
			if (fcount < BATCH_SIZE && f + 1 != (int)file_names.size()) continue;//\B1\A3֤\CB\CD\C8\EB\CD\C6\C0\ED\B5\C4Ϊһ\B8\F6batchsize,fcount = batchsize,
			for (int b = 0; b < fcount; b++) {//b\D7\EE\B4\F3Ϊbatchsize-1
				cv::Mat img = cv::imread(std::string(img_path) + "/" + file_names[f - fcount + 1 + b]); //\B5\DAһ\D5\C5ͼΪ7-8+1+0=0
				if (img.empty()) continue;
				cv::Mat pr_img = preprocess_img(img); // letterbox BGR to RGB
				int i = 0;
				for (int row = 0; row < INPUT_H; ++row) {
					uchar* uc_pixel = pr_img.data + row * pr_img.step;//\B5\DAn\D0\D0\CA\D7Ԫ\CBصĵ\D8ַ
					for (int col = 0; col < INPUT_W; ++col) {//\CF\F1\CBع\E9һ\BB\AF
						data[b * 3 * INPUT_H * INPUT_W + i] = (float)uc_pixel[2] / 255.0;
						data[b * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
						data[b * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
						uc_pixel += 3;
						++i;
					}
				}
			}
			// Run inference
			auto start = std::chrono::system_clock::now();
			doInference(*context, stream, buffers, data, prob, BATCH_SIZE);
			auto end = std::chrono::system_clock::now();
			std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
			std::vector<std::vector<Yolo::Detection>> batch_res(fcount);
			for (int b = 0; b < fcount; b++) {
				auto& res = batch_res[b];
				nms(res, &prob[b * OUTPUT_SIZE], CONF_THRESH, NMS_THRESH);
			}
			for (int b = 0; b < fcount; b++) {
				auto& res = batch_res[b];
				//std::cout << res.size() << std::endl;
				cv::Mat img = cv::imread(std::string(img_path) + "/" + file_names[f - fcount + 1 + b]);
				for (size_t j = 0; j < res.size(); j++) {
					cv::Rect r = get_rect(img, res[j].bbox);
					cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
					cv::putText(img, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
				}
				cv::imwrite("_" + file_names[f - fcount + 1 + b], img);
			}
			fcount = 0;
		}
	// Release stream and buffers
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex]));
	CHECK(cudaFree(buffers[outputIndex]));
	// Destroy the engine
	context->destroy();
	engine->destroy();
	runtime->destroy();

	}
	else if (pattern == 2) {
        //cam_initialize();
		// prepare input data ---------------------------
		static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
		//for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
		//    data[i] = 1.0;
		static float prob[BATCH_SIZE * OUTPUT_SIZE];
		IRuntime* runtime = createInferRuntime(gLogger);
		assert(runtime != nullptr);
		ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
		assert(engine != nullptr);
		IExecutionContext* context = engine->createExecutionContext();
		assert(context != nullptr);
		delete[] trtModelStream;
		assert(engine->getNbBindings() == 2);
		void* buffers[2];
		// In order to bind the buffers, we need to know the names of the input and output tensors.
		// Note that indices are guaranteed to be less than IEngine::getNbBindings()
		const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
		const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
		assert(inputIndex == 0);
		assert(outputIndex == 1);
		// Create GPU buffers on device
		CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
		CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
		// Create stream
		cudaStream_t stream;
		CHECK(cudaStreamCreate(&stream));
        
		//cv::VideoCapture cap;
		//cap.open(img_path);
		//cv::Mat img;
        /*
		if (!cap.isOpened()){
			std::cout << "Video open failed" << std::endl;
			return;
		}
        */
        cv::Mat img;
        FlyCapture2::Error error;
        Image image;
        Image image2;
        Mat cameraMatrix = Mat::eye(3,3,CV_64F);
        Mat distCoeffs=Mat::zeros(5,1,CV_64F);
        Mat view,rview,map1,map2,dst;
        Size imageSize;
/*
        Ptr<Tracker> tracker;
        tracker=TrackerKCF::create();
        Rect2d bbox;
*/
        while (1) {
            //cap >> img;
            error = pCameras[0].RetrieveBuffer(&image);
            if (error != PGRERROR_OK)
            {
                PrintError(error);
                delete[] pCameras;
                std::cout << "Press Enter to exit." << endl;
                cin.ignore();
            }
            error = image.Convert(PIXEL_FORMAT_BGR, &image2);

            unsigned int rowBytes = (double)image2.GetDataSize() / (double)image2.GetRows();
            //图像传输
            cv::Mat src = cv::Mat(image2.GetRows(), image2.GetCols(), CV_8UC3, image2.GetData(), rowBytes);
            //src=imread("../1.jpg");
            //cap>>src;
            resize(src,src,Size(1024,1024));
            /*
            Mat frameCalibration;
            cameraMatrix.at<double>(0, 0) = 6.355809472341207e+02;
            cameraMatrix.at<double>(0, 1) = 0;
            cameraMatrix.at<double>(0, 2) = 5.048749733034192e+02;
            cameraMatrix.at<double>(1, 1) = 6.356741995280925e+02;
            cameraMatrix.at<double>(1, 2) = 4.825381259048260e+02;
            cameraMatrix.at<double>(2, 2) = 1;

            distCoeffs.at<double>(0, 0) = -0.264788000153760;
            distCoeffs.at<double>(1, 0) = 0.046394008673741;
            distCoeffs.at<double>(2, 0) = 0.004060476522896;
            distCoeffs.at<double>(3, 0) = 3.429612389369573e-04;
            distCoeffs.at<double>(4, 0) = 0;
            imageSize = Size(1024, 1024);
            initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
                getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0),
                imageSize, CV_16SC2, map1, map2);
            remap(src, frameCalibration, map1, map2, INTER_LINEAR);
            resize(frameCalibration, frameCalibration, Size(src.rows * 0.8, src.cols * 0.8));
            frameCalibration.copyTo(dst);
            */
            //imshow("src", frameCalibration);
            
            resize(src,src,Size(1024,1024));
            imshow("src",src);
            setMouseCallback("src",onMouse1,&src);
            
            //bbox=selectROI(src,false);
            //tracker->init(src,bbox);
            //setMouseCallback("src", onMouse1, &frameCalibration);
            char c = waitKey();
            //标定完成，进入相机流
            if (c =='q') {
                destroyWindow("src");
                break;
            };
        }

        
        KF_1_1.transitionMatrix =(Mat_<float>(4, 4) << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1);  //转移矩阵A
        setIdentity(KF_1_1.measurementMatrix);                                             //测量矩阵H
        setIdentity(KF_1_1.processNoiseCov, Scalar::all(1e-3));                            //系统噪声方差矩阵Q
        setIdentity(KF_1_1.measurementNoiseCov, Scalar::all(1e-1));                        //测量噪声方差矩阵R
        setIdentity(KF_1_1.errorCovPost, Scalar::all(1));                                  //后验错误估计协方差矩阵P
        rng.fill(KF_1_1.statePost, RNG::UNIFORM, 0, 1024);   //初始状态值x(0)

        KF_1_2.transitionMatrix =(Mat_<float>(4, 4) << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1);  //转移矩阵A
        setIdentity(KF_1_2.measurementMatrix);                                             //测量矩阵H
        setIdentity(KF_1_2.processNoiseCov, Scalar::all(1e-3));                            //系统噪声方差矩阵Q
        setIdentity(KF_1_2.measurementNoiseCov, Scalar::all(1e-1));                        //测量噪声方差矩阵R
        setIdentity(KF_1_2.errorCovPost, Scalar::all(1));                                  //后验错误估计协方差矩阵P
        rng.fill(KF_1_2.statePost, RNG::UNIFORM, 0, 1024); 

        int t=0;
        int t1=0;
        Mat out;
        double last_1_x=0;
        double last_1_y=0;
        double last_2_x=0;
        double last_2_y=0;
        double last_3_x=0;
        double last_3_y=0;
        double last_4_x=0;
        double last_4_y=0;
        double distance1;
        double distance2;

        int last_blue=0;
        int last_red=0;

        bool loss=false;
        while(1){

            error = pCameras[0].RetrieveBuffer(&image);

            if (error != PGRERROR_OK)
            {
                PrintError(error);
                delete[] pCameras;
                std::cout << "Press Enter to exit." << endl;
                cin.ignore();
            }

            error = image.Convert(PIXEL_FORMAT_BGR, &image2);

            unsigned int rowBytes = (double)image2.GetDataSize() / (double)image2.GetRows();
            error = image.Convert(PIXEL_FORMAT_BGR, &image2);
            
            //Mat src;
            cv::Mat src = cv::Mat(image2.GetRows(), image2.GetCols(), CV_8UC3, image2.GetData(), rowBytes);
            //cv::Mat src=imread("../3208.jpg");
            //cap>>src;

			if (src.empty()) break;
			cv::Mat dst = src.clone();
            resize(dst,dst,Size(1024,1024));

			cv::Mat pr_img = preprocess_img(dst);
			int i = 0;
            int blue=0;
            int red=0;
            int true_blue=0;

			for (int row = 0; row < INPUT_H; ++row) {
				uchar* uc_pixel = pr_img.data + row * pr_img.step;//\B5\DAn\D0\D0\CA\D7Ԫ\CBصĵ\D8ַ
				for (int col = 0; col < INPUT_W; ++col) {//\CF\F1\CBع\E9һ\BB\AF
					data[0 * 3 * INPUT_H * INPUT_W + i] = (float)uc_pixel[2] / 255.0;
					data[0 * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
					data[0 * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
					uc_pixel += 3;
					++i;
				}
			}
			auto start = std::chrono::system_clock::now();
			doInference(*context, stream, buffers, data, prob, BATCH_SIZE);
			auto end = std::chrono::system_clock::now();
			std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
			std::vector<std::vector<Yolo::Detection>> batch_res(1);
			for (int b = 0; b < 1; b++) {
				auto& res = batch_res[b];
				nms(res, &prob[b * OUTPUT_SIZE], CONF_THRESH, NMS_THRESH);
			}          
            
            warpPerspective(dst,out,mask,Size(808,448));
			for (int b = 0; b < 1; b++) {
				auto& res = batch_res[b];
				//std::cout << res.size() << std::endl;
				//cv::Mat img = cv::imread(std::string(img_path) + "/" + file_names[f - fcount + 1 + b]);
				for (size_t j = 0; j < res.size(); j++) {
        
					cv::Rect r = get_rect(dst, res[j].bbox);
					//cv::rectangle(dst, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
					//cv::putText(dst, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
                    double centerx=r.x+r.width/2;
                    double centery=r.y+r.height/2;
                    org1=(Mat_<double>(3,1)<<centerx,centery,1);
                    res1=mask*org1;
                    

                    if ((int)res[j].class_id==0)
                    {
                        if(blue==0)
                        {
                            //circle(dst,Point(centerx,centery),7,Scalar(255,0,0),5);
                            
                            post_1_blue_1x=*res1.ptr<double>(0,0)/(*res1.ptr<double>(2,0));
                            post_1_blue_1y=*res1.ptr<double>(1,0)/(*res1.ptr<double>(2,0));
                            
                            cout<<"car1 god: "<<post_1_blue_1x<<" "<<post_1_blue_1y<<endl;
                            
                            if(post_1_blue_1x<808&&post_1_blue_1y<448)
                            {
                                blue++;
                                true_blue++;
                                //b1=true;
                            }
                            else
                            {
                               //post_1_blue_1x=0;
                                //post_1_blue_1y=0;
                                blue++;
                                //b1=false;
                            }
                            
                        }
                        else if(blue==1)
                        {
                            //circle(dst,Point(centerx,centery),7,Scalar(0,255,0),5);
                            post_1_blue_2x=*res1.ptr<double>(0,0)/(*res1.ptr<double>(2,0));
                            post_1_blue_2y=*res1.ptr<double>(1,0)/(*res1.ptr<double>(2,0));
                            
                            cout<<"car2 god: "<<post_1_blue_2x<<" "<<post_1_blue_2y<<endl;
                            if(post_1_blue_2x<808&&post_1_blue_2y<448)
                            {
                                blue++;
                                //true_blue++;
                                //b2=true;
                            }
                            else
                            {
                                //post_1_blue_2x=0;
                                //post_1_blue_2y=0;
                                blue++;
                                //b2=false;
                            }
                        }
                    }
                    else
                    {
                        post_1_red_1x=*res1.ptr<double>(0,0)/(*res1.ptr<double>(2,0));
                        post_1_red_1y=*res1.ptr<double>(1,0)/(*res1.ptr<double>(2,0));
                        //cout<<"god: "<<post_1_red_1x<<" "<<post_1_red_1y<<endl;
                        circle(out,Point(post_1_red_1x,post_1_red_1y),5,Scalar(0,255,0),5);
                    }

                    
                }
				//cv::imwrite("dst.jpg", dst);
                if(t!=0)
                {
                    distance1=sqrt((post_1_blue_1x-last_1_x)*(post_1_blue_1x-last_1_x)+(post_1_blue_1y-last_1_y)*(post_1_blue_1y-last_1_y));
                    distance2=sqrt((post_1_blue_2x-last_2_x)*(post_1_blue_2x-last_2_x)+(post_1_blue_2y-last_2_y)*(post_1_blue_2y-last_2_y));
                    cout<<"distance1: "<<distance1<<"  distance2: "<<distance2<<endl;
                    if (t1==0||t1==1)
                    {
                        t1++;
                    }
                    /*
                    if((distance1>15||distance2>15)&&t1!=1)
                    {
                        
                        if(blue==1)
                        {
                            if(distance1>distance2)
                            {
                                post_1_blue_2x=post_1_blue_1x;
                                post_1_blue_2y=post_1_blue_1y;

                                post_1_blue_1x=0;
                                post_1_blue_1y=0;

                            }
                            else{
                                post_1_blue_2x=0;
                                post_1_blue_2y=0;
                            }
                            //loss=true;
                            
                        }
                        if(blue==2)                        
                        {
                            //if(!loss)
                            {
                                swap(post_1_blue_1x,post_1_blue_2x);
                                swap(post_1_blue_1y,post_1_blue_2y);
                                cout<<"**********change**********"<<endl;
                            }
                            //loss=false;
                        }
                    }
                    */
                    
                    if(t1!=1)
                    {
                        if(blue==1)
                        {
                            double d1=calculate_distance(post_1_blue_1x,post_1_blue_1y,last_1_x,last_1_y);
                            double d2=calculate_distance(post_1_blue_1x,post_1_blue_1y,last_2_x,last_2_y);    
                            cout<<"d1: "<<d1<<"d2:  "<<d2<<endl;    
                                if(d1>d2)
                                {
                                    post_1_blue_2x=post_1_blue_1x;
                                    post_1_blue_2y=post_1_blue_1y;

                                    post_1_blue_1x=0;
                                    post_1_blue_1y=0;

                                }
                                else{
                                    post_1_blue_2x=0;
                                    post_1_blue_2y=0;
                                }
                        }
                        else if(blue==2)
                        {
                            if((distance1>30||distance2>30)&&t1!=1)
                            {
                                if(min(distance1,distance2)<30)
                                {
                                    
                                }
                                else{
                                    swap(post_1_blue_1x,post_1_blue_2x);
                                    swap(post_1_blue_1y,post_1_blue_2y);
                                    cout<<"**********change**********"<<endl;
                                }
                            }
                        }
                    } 
                    
                    cout<<"final car1 "<<post_1_blue_1x<<" "<<post_1_blue_1y<<endl;
                    cout<<"final car2 "<<post_1_blue_2x<<" "<<post_1_blue_2y<<endl;
                }
                else{t++;}
                
/*
                bool ok = tracker->update(dst,bbox);
                if(ok)
                {
                    rectangle(dst,bbox,Scalar(255, 0, 0), 2, 1);
                }

                else{
                    putText(dst,"failure",Point(100, 80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
                }
*/

                if(post_1_blue_1x<808&&post_1_blue_1y<448)
                {
                    circle(out,Point(post_1_blue_1x,post_1_blue_1y),5,Scalar(255,0,0),5);
                }
                if(post_1_blue_2x<808&&post_1_blue_2y<448)
                {
                    circle(out,Point(post_1_blue_2x,post_1_blue_2y),5,Scalar(255,0,0),5);
                }
                last_1_x=post_1_blue_1x;
                last_1_y=post_1_blue_1y;
                last_2_x=post_1_blue_2x;
                last_2_y=post_1_blue_2y;
                
                if(post_1_blue_1x!=0&&post_1_blue_1y!=0)
                {
                    kalman(post_1_blue_1x,post_1_blue_1y);
                    circle(out,Point(kalman_1_1x,kalman_1_1y),5,Scalar(0,0,255),5);
                }
                //circle(out,Point(temp1,temp2),5,Scalar(0,0,255),5);
                if(post_1_blue_2x!=0&&post_1_blue_2y!=0)
                {
                    kalman_1_2(post_1_blue_2x,post_1_blue_2y);
                    circle(out,Point(kalman_1_2x,kalman_1_2y),5,Scalar(0,0,255),5);
                }

				cv::imshow("detect1", dst);
				cv::imshow("out1",out);
			}
			if (cv::waitKey(20) == 27)
				//destroyAllWindows();
                break;
		}
		//cap.release();
	// Release stream and buffers
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex]));
	CHECK(cudaFree(buffers[outputIndex]));
	// Destroy the engine
	context->destroy();
	engine->destroy();
	runtime->destroy();
	}
}

void infergai1(int pattern,const char *img_path,std::string engine_name)
{
    printf("thread2");
    cudaSetDevice(DEVICE1);
	// create a model using the API directly and serialize it to a stream
	char *trtModelStream{ nullptr };
	size_t size{ 0 };

	std::ifstream file(engine_name, std::ios::binary);
	if (file.good()) {
		file.seekg(0, file.end);
		size = file.tellg();
		file.seekg(0, file.beg);
		trtModelStream = new char[size];
		assert(trtModelStream);
		file.read(trtModelStream, size);
		file.close();
	}
	if (pattern == 1){
		std::vector<std::string> file_names;
		if (read_files_in_dir(img_path, file_names) < 0) {
			std::cout << "read_files_in_dir failed." << std::endl;
			return;
		}
		// prepare input data ---------------------------
		static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
		//for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
		//    data[i] = 1.0;
		static float prob[BATCH_SIZE * OUTPUT_SIZE];
		IRuntime* runtime = createInferRuntime(gLogger);
		assert(runtime != nullptr);
		ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
		assert(engine != nullptr);
		IExecutionContext* context = engine->createExecutionContext();
		assert(context != nullptr);
		delete[] trtModelStream;
		assert(engine->getNbBindings() == 2);
		void* buffers[2];
		// In order to bind the buffers, we need to know the names of the input and output tensors.
		// Note that indices are guaranteed to be less than IEngine::getNbBindings()
		const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
		const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
		assert(inputIndex == 0);
		assert(outputIndex == 1);
		// Create GPU buffers on device
		CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
		CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
		// Create stream
		cudaStream_t stream;
		CHECK(cudaStreamCreate(&stream));
		int fcount = 0;
		for (int f = 0; f < (int)file_names.size(); f++) { //fΪ\CEļ\FE\BC\D0\D6\D0ͼƬ\B5\C4index,Ϊfcount\B5ı\B6\CA\FD
			fcount++;
			if (fcount < BATCH_SIZE && f + 1 != (int)file_names.size()) continue;//\B1\A3֤\CB\CD\C8\EB\CD\C6\C0\ED\B5\C4Ϊһ\B8\F6batchsize,fcount = batchsize,
			for (int b = 0; b < fcount; b++) {//b\D7\EE\B4\F3Ϊbatchsize-1
				cv::Mat img = cv::imread(std::string(img_path) + "/" + file_names[f - fcount + 1 + b]); //\B5\DAһ\D5\C5ͼΪ7-8+1+0=0
				if (img.empty()) continue;
				cv::Mat pr_img = preprocess_img(img); // letterbox BGR to RGB
				int i = 0;
				for (int row = 0; row < INPUT_H; ++row) {
					uchar* uc_pixel = pr_img.data + row * pr_img.step;//\B5\DAn\D0\D0\CA\D7Ԫ\CBصĵ\D8ַ
					for (int col = 0; col < INPUT_W; ++col) {//\CF\F1\CBع\E9һ\BB\AF
						data[b * 3 * INPUT_H * INPUT_W + i] = (float)uc_pixel[2] / 255.0;
						data[b * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
						data[b * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
						uc_pixel += 3;
						++i;
					}
				}
			}
			// Run inference
			auto start = std::chrono::system_clock::now();
			doInference(*context, stream, buffers, data, prob, BATCH_SIZE);
			auto end = std::chrono::system_clock::now();
			std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
			std::vector<std::vector<Yolo::Detection>> batch_res(fcount);
			for (int b = 0; b < fcount; b++) {
				auto& res = batch_res[b];
				nms(res, &prob[b * OUTPUT_SIZE], CONF_THRESH, NMS_THRESH);
			}
			for (int b = 0; b < fcount; b++) {
				auto& res = batch_res[b];
				//std::cout << res.size() << std::endl;
				cv::Mat img = cv::imread(std::string(img_path) + "/" + file_names[f - fcount + 1 + b]);
				for (size_t j = 0; j < res.size(); j++) {
					cv::Rect r = get_rect(img, res[j].bbox);
					cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
					cv::putText(img, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
				}
				cv::imwrite("_" + file_names[f - fcount + 1 + b], img);
			}
			fcount = 0;
		}
	// Release stream and buffers
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex]));
	CHECK(cudaFree(buffers[outputIndex]));
	// Destroy the engine
	context->destroy();
	engine->destroy();
	runtime->destroy();

	}
	else if (pattern == 2) {
        //cam_initialize();
		// prepare input data ---------------------------
		static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
		//for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
		//    data[i] = 1.0;
		static float prob[BATCH_SIZE * OUTPUT_SIZE];
		IRuntime* runtime = createInferRuntime(gLogger);
		assert(runtime != nullptr);
		ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
		assert(engine != nullptr);
		IExecutionContext* context = engine->createExecutionContext();
		assert(context != nullptr);
		delete[] trtModelStream;
		assert(engine->getNbBindings() == 2);
		void* buffers[2];
		// In order to bind the buffers, we need to know the names of the input and output tensors.
		// Note that indices are guaranteed to be less than IEngine::getNbBindings()
		const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
		const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
		assert(inputIndex == 0);
		assert(outputIndex == 1);
		// Create GPU buffers on device
		CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
		CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
		// Create stream
		cudaStream_t stream;
		CHECK(cudaStreamCreate(&stream));
        
		cv::VideoCapture cap;
		cap.open(img_path);
		//cv::Mat img;
        
		if (!cap.isOpened()){
			std::cout << "Video open failed" << std::endl;
			return;
		}
        
        cv::Mat img;
        FlyCapture2::Error error;
        Image image;
        Image image2;
        Mat cameraMatrix = Mat::eye(3,3,CV_64F);
        Mat distCoeffs=Mat::zeros(5,1,CV_64F);
        Mat view,rview,map1,map2,dst;
        Size imageSize;

        while (1) {
            //cap >> img;
            error = pCameras[0].RetrieveBuffer(&image);
            if (error != PGRERROR_OK)
            {
                PrintError(error);
                delete[] pCameras;
                std::cout << "Press Enter to exit." << endl;
                cin.ignore();
            }

            error = image.Convert(PIXEL_FORMAT_BGR, &image2);

            unsigned int rowBytes = (double)image2.GetDataSize() / (double)image2.GetRows();
            //图像传输
            //cv::Mat src = cv::Mat(image2.GetRows(), image2.GetCols(), CV_8UC3, image2.GetData(), rowBytes);
            Mat src;
            cap>>src;
            resize(src,src,Size(1024,1024));
            /*
            Mat frameCalibration;
            cameraMatrix.at<double>(0, 0) = 6.355809472341207e+02;
            cameraMatrix.at<double>(0, 1) = 0;
            cameraMatrix.at<double>(0, 2) = 5.048749733034192e+02;
            cameraMatrix.at<double>(1, 1) = 6.356741995280925e+02;
            cameraMatrix.at<double>(1, 2) = 4.825381259048260e+02;
            cameraMatrix.at<double>(2, 2) = 1;

            distCoeffs.at<double>(0, 0) = -0.264788000153760;
            distCoeffs.at<double>(1, 0) = 0.046394008673741;
            distCoeffs.at<double>(2, 0) = 0.004060476522896;
            distCoeffs.at<double>(3, 0) = 3.429612389369573e-04;
            distCoeffs.at<double>(4, 0) = 0;
            imageSize = Size(1024, 1024);
            initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
                getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0),
                imageSize, CV_16SC2, map1, map2);
            remap(src, frameCalibration, map1, map2, INTER_LINEAR);
            resize(frameCalibration, frameCalibration, Size(src.rows, src.cols));
            frameCalibration.copyTo(dst);
            
            imshow("src1", frameCalibration);
*/
            imshow("src1",src);
            setMouseCallback("src1", onMouse2, &src);
            char c = waitKey();
            //标定完成，进入相机流
            if (c == 'q') {
                destroyWindow("src1");
                break;
            };
        }

        int t=0;
        int t1=0;
        Mat out;
        double last_1_x=0;
        double last_1_y=0;
        double last_2_x=0;
        double last_2_y=0;
        double last_3_x;
        double last_3_y;
        double last_4_x;
        double last_4_y;
        double distance1;
        double distance2;

        int last_blue=0;
        int last_red=0;

        KF_2_1.transitionMatrix =(Mat_<float>(4, 4) << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1);  //转移矩阵A
        setIdentity(KF_2_1.measurementMatrix);                                             //测量矩阵H
        setIdentity(KF_2_1.processNoiseCov, Scalar::all(1e-3));                            //系统噪声方差矩阵Q
        setIdentity(KF_2_1.measurementNoiseCov, Scalar::all(1e-1));                        //测量噪声方差矩阵R
        setIdentity(KF_2_1.errorCovPost, Scalar::all(1));                                  //后验错误估计协方差矩阵P
        rng.fill(KF_2_1.statePost, RNG::UNIFORM, 0, 1024);   //初始状态值x(0)

        KF_2_2.transitionMatrix =(Mat_<float>(4, 4) << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1);  //转移矩阵A
        setIdentity(KF_2_2.measurementMatrix);                                             //测量矩阵H
        setIdentity(KF_2_2.processNoiseCov, Scalar::all(1e-3));                            //系统噪声方差矩阵Q
        setIdentity(KF_2_2.measurementNoiseCov, Scalar::all(1e-1));                        //测量噪声方差矩阵R
        setIdentity(KF_2_2.errorCovPost, Scalar::all(1));                                  //后验错误估计协方差矩阵P
        rng.fill(KF_2_2.statePost, RNG::UNIFORM, 0, 1024);                                          //测量矩阵H
        

        while(1){

            error = pCameras[0].RetrieveBuffer(&image);

            if (error != PGRERROR_OK)
            {
                PrintError(error);
                delete[] pCameras;
                std::cout << "Press Enter to exit." << endl;
                cin.ignore();
            }

            error = image.Convert(PIXEL_FORMAT_BGR, &image2);

            unsigned int rowBytes = (double)image2.GetDataSize() / (double)image2.GetRows();
            error = image.Convert(PIXEL_FORMAT_BGR, &image2);
            Mat src;
            //cv::Mat src = cv::Mat(image2.GetRows(), image2.GetCols(), CV_8UC3, image2.GetData(), rowBytes);
            cap>>src;
			if (src.empty()) break;
			cv::Mat dst = src.clone();
            resize(dst,dst,Size(1024,1024));
			cv::Mat pr_img = preprocess_img(dst);
			int i = 0;
            int blue=0;
            int red=0;

			for (int row = 0; row < INPUT_H; ++row) {
				uchar* uc_pixel = pr_img.data + row * pr_img.step;//\B5\DAn\D0\D0\CA\D7Ԫ\CBصĵ\D8ַ
				for (int col = 0; col < INPUT_W; ++col) {//\CF\F1\CBع\E9һ\BB\AF
					data[0 * 3 * INPUT_H * INPUT_W + i] = (float)uc_pixel[2] / 255.0;
					data[0 * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
					data[0 * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
					uc_pixel += 3;
					++i;
				}
			}
			auto start = std::chrono::system_clock::now();
			doInference(*context, stream, buffers, data, prob, BATCH_SIZE);
			auto end = std::chrono::system_clock::now();
			std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
			std::vector<std::vector<Yolo::Detection>> batch_res(1);
			for (int b = 0; b < 1; b++) {
				auto& res = batch_res[b];
				nms(res, &prob[b * OUTPUT_SIZE], CONF_THRESH, NMS_THRESH);
			}
            warpPerspective(dst,out,mask2,Size(808,448));
			for (int b = 0; b < 1; b++) {
				auto& res = batch_res[b];
				//std::cout << res.size() << std::endl;
				//cv::Mat img = cv::imread(std::string(img_path) + "/" + file_names[f - fcount + 1 + b]);
				for (size_t j = 0; j < res.size(); j++) {
					cv::Rect r = get_rect(dst, res[j].bbox);
					cv::rectangle(dst, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
					cv::putText(dst, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
				
                    double centerx=r.x+r.width/2;
                    double centery=r.y+r.height/2;
                    org2=(Mat_<double>(3,1)<<centerx,centery,1);
                    res2=mask2*org2;


/*
                    if ((int)res[j].class_id==0)
                    {
                        post_2_blue_1x=*res2.ptr<double>(0,0)/(*res2.ptr<double>(2,0));
                        post_2_blue_1y=*res2.ptr<double>(1,0)/(*res2.ptr<double>(2,0));
                    }
                    else
                    {
                        post_2_red_1x=*res2.ptr<double>(0,0)/(*res2.ptr<double>(2,0));
                        post_2_red_1y=*res2.ptr<double>(1,0)/(*res2.ptr<double>(2,0));
                    }
*/
                    if ((int)res[j].class_id==0)
                    {
                        if(blue==0)
                        {                        
                            post_2_blue_1x=*res2.ptr<double>(0,0)/(*res2.ptr<double>(2,0));
                            post_2_blue_1y=*res2.ptr<double>(1,0)/(*res2.ptr<double>(2,0));
                            
                            cout<<"car1 god: "<<post_2_blue_1x<<" "<<post_2_blue_1y<<endl;
                            
                            if(post_2_blue_1x<808&&post_2_blue_1y<448)
                            {
                                blue++;
                            }
                            else
                            {
                                blue++;
                            }
                            
                        }
                        else if(blue==1)
                        {
                            post_2_blue_2x=*res2.ptr<double>(0,0)/(*res2.ptr<double>(2,0));
                            post_2_blue_2y=*res2.ptr<double>(1,0)/(*res2.ptr<double>(2,0));                            
                            cout<<"car2 god: "<<post_2_blue_2x<<" "<<post_2_blue_2y<<endl;
                            if(post_2_blue_2x<808&&post_2_blue_2y<448)
                            {
                                blue++;
                            }
                            else
                            {
                                blue++;
                            }
                        }
                    }
                    else
                    {
                        post_2_red_1x=*res2.ptr<double>(0,0)/(*res2.ptr<double>(2,0));
                        post_2_red_1y=*res2.ptr<double>(1,0)/(*res2.ptr<double>(2,0));
                        //cout<<"god: "<<post_1_red_1x<<" "<<post_1_red_1y<<endl;
                        circle(out,Point(post_2_red_1x,post_2_red_1y),5,Scalar(0,255,0),5);
                    }

                
                }

                if(t!=0)
                {
                    distance1=sqrt((post_2_blue_1x-last_1_x)*(post_2_blue_1x-last_1_x)+(post_2_blue_1y-last_1_y)*(post_2_blue_1y-last_1_y));
                    distance2=sqrt((post_2_blue_2x-last_2_x)*(post_2_blue_2x-last_2_x)+(post_2_blue_2y-last_2_y)*(post_2_blue_2y-last_2_y));
                    cout<<"distance1: "<<distance1<<"  distance2: "<<distance2<<endl;
                    if (t1==0||t1==1)
                    {
                        t1++;
                    }
                    /*
                    if((distance1>15||distance2>15)&&t1!=1)
                    {
                        
                        if(blue==1)
                        {
                            if(distance1>distance2)
                            {
                                post_1_blue_2x=post_1_blue_1x;
                                post_1_blue_2y=post_1_blue_1y;

                                post_1_blue_1x=0;
                                post_1_blue_1y=0;

                            }
                            else{
                                post_1_blue_2x=0;
                                post_1_blue_2y=0;
                            }
                            //loss=true;
                            
                        }
                        if(blue==2)                        
                        {
                            //if(!loss)
                            {
                                swap(post_1_blue_1x,post_1_blue_2x);
                                swap(post_1_blue_1y,post_1_blue_2y);
                                cout<<"**********change**********"<<endl;
                            }
                            //loss=false;
                        }
                    }
                    */
                    
                    if(t1!=1)
                    {
                        if(blue==1)
                        {
                            double d1=calculate_distance(post_2_blue_1x,post_2_blue_1y,last_1_x,last_1_y);
                            double d2=calculate_distance(post_2_blue_1x,post_2_blue_1y,last_2_x,last_2_y);    
                            cout<<"d1: "<<d1<<"d2:  "<<d2<<endl;    
                                if(d1>d2)
                                {
                                    post_2_blue_2x=post_2_blue_1x;
                                    post_2_blue_2y=post_2_blue_1y;

                                    post_2_blue_1x=0;
                                    post_2_blue_1y=0;

                                }
                                else{
                                    post_2_blue_2x=0;
                                    post_2_blue_2y=0;
                                }
                        }
                        else if(blue==2)
                        {
                            if((distance1>30||distance2>30)&&t1!=1)
                            {
                                if(min(distance1,distance2)>30)
                                {
                                    swap(post_2_blue_1x,post_2_blue_2x);
                                    swap(post_2_blue_1y,post_2_blue_2y);
                                    cout<<"**********change**********"<<endl;
                                }
                            }
                        }
                    } 
                    
                    cout<<"final car1 "<<post_2_blue_1x<<" "<<post_2_blue_1y<<endl;
                    cout<<"final car2 "<<post_2_blue_2x<<" "<<post_2_blue_2y<<endl;
                }
                else{t++;}
                /*
                bool ok = tracker->update(dst,bbox);

                if(ok)
                {
                    rectangle(dst,bbox,Scalar(255, 0, 0), 2, 1);
                }
                else{
                    putText(dst,"failure",Point(100, 80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
                }
                */
                if(post_2_blue_1x<808&&post_2_blue_1y<448)
                {
                    circle(out,Point(post_2_blue_1x,post_2_blue_1y),5,Scalar(255,0,0),5);
                }
                if(post_2_blue_2x<808&&post_2_blue_2y<448)
                {
                    circle(out,Point(post_2_blue_2x,post_2_blue_2y),5,Scalar(255,0,0),5);
                }
                last_1_x=post_2_blue_1x;
                last_1_y=post_2_blue_1y;
                last_2_x=post_2_blue_2x;
                last_2_y=post_2_blue_2y;
                
                if(post_2_blue_1x!=0&&post_2_blue_1y!=0)
                {
                    kalman_2_1(post_2_blue_1x,post_2_blue_1y);
                    circle(out,Point(kalman_2_1x,kalman_2_1y),5,Scalar(0,0,255),5);
                }
                if(post_2_blue_2x!=0&&post_2_blue_2y!=0)
                {
                    kalman_2_2(post_2_blue_2x,post_2_blue_2y);
                    circle(out,Point(kalman_2_2x,kalman_2_2y),5,Scalar(0,0,255),5);
                }

				cv::imshow("out2",out);
				cv::imshow("detect2", dst);
				
			}
			if (cv::waitKey(20) == 27)
            {
				destroyAllWindows();
                break;
            }
		}
		//cap.release();
	// Release stream and buffers
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex]));
	CHECK(cudaFree(buffers[outputIndex]));
	// Destroy the engine
	context->destroy();
	engine->destroy();
	runtime->destroy();
	}  
}

/*
char recvline[MAXLINE];
char sendline[MAXLINE];
int serverfd;
struct sockaddr_in serveraddr , clientaddr;
int send_length = 0;
unsigned int server_addr_length, client_addr_length;
*/
int sock_fd,client_fd;
int ret;
struct sockaddr_in localaddr;
struct sockaddr_in recvaddr;
socklen_t  socklen;
char recv_buf[100];
char send_buf[100];
int ttl = 10;//如果转发的次数等于10,则不再转发
int loop=0;
unsigned int local_addr_length, recv_addr_length;

int sock_fd1,client_fd1;
int ret1;
struct sockaddr_in localaddr1;
struct sockaddr_in recvaddr1;
socklen_t  socklen1;
char recv_buf1[100];
char send_buf1[100];
int ttl1 = 10;//如果转发的次数等于10,则不再转发
int loop1=0;
unsigned int local_addr_length1, recv_addr_length1;

void zcx()
{
    cout<<"************************zcx start********************************************"<<endl;
    if( (sock_fd = socket(AF_INET, SOCK_DGRAM, 0)) < 0 ){
         perror("socket() error");
         exit(1);
     }
    
    cout<<"+++++++++++++++++++++++2+++++++++++++++++++++++++++++"<<endl;

    socklen = sizeof(struct sockaddr);                

     // 通过struct sockaddr_in 结构设置服务器地址和监听端口；
     bzero(&localaddr,sizeof(sock_fd));
     //bzero(&clientaddr,sizeof(clientaddr));
     localaddr.sin_family = AF_INET;
     localaddr.sin_addr.s_addr = htonl(INADDR_ANY);
     localaddr.sin_port = htons(6666);//htons(UDPPORT);
     local_addr_length = sizeof(localaddr);

     // 使用bind() 函数绑定监听端口，将套接字文件描述符和地址类型变量（struct sockaddr_in ）进行绑定；
     if( bind(sock_fd, (struct sockaddr *) &localaddr, local_addr_length) < 0){
         perror("bind() error");
         exit(1); 
     }
     cout<<"+++++++++++++++++++++++3+++++++++++++++++++++++++++++"<<endl;
    
     
       //设置多播的TTL值
        if(setsockopt(sock_fd, IPPROTO_IP, IP_MULTICAST_TTL, &ttl, sizeof(ttl))<0){
                perror("IP_MULTICAST_TTL");
                //return -1;
        }
        //设置数据是否发送到本地回环接口
        if(setsockopt(sock_fd, IPPROTO_IP, IP_MULTICAST_LOOP, &loop, sizeof(loop))<0){
                perror("IP_MULTICAST_LOOP");
                //return -1;
        }
        //加入多播组
        struct ip_mreq mreq;
        mreq.imr_multiaddr.s_addr=inet_addr(MCAST);//多播组的IP
        mreq.imr_interface.s_addr=htonl(INADDR_ANY);//本机的默认接口IP,本机的随机IP
        if(setsockopt(sock_fd, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq)) < 0){
                perror("IP_ADD_MEMBERSHIP");
                //return -1;
        }
     // 接收客户端的数据，使用recvfrom() 函数接收客户端的网络数据；
     recv_addr_length = sizeof(sockaddr_in);
    
    cout<<"+++++++++++++++++++++++4+++++++++++++++++++++++++++++"<<endl;
    
     int recv_length = 0;
     recv_length = recvfrom(sock_fd, recv_buf, sizeof(recv_buf), 0, (struct sockaddr *) &recvaddr, &socklen);
     cout << "recv_length = "<< recv_length <<endl;
     cout << recv_buf << endl;

    cout<<"+++++++++++++++++++++++5+++++++++++++++++++++++++++++"<<endl;

     recv_addr_length = sizeof(sockaddr_in);
    while(1)
    {

        
        
            if(post_1_blue_1x!=0&&post_1_blue_1y!=0&&post_2_blue_1x!=0&&post_2_blue_1y!=0)
            {
                sgdata.car1=true;
                sgdata.car1_x=(post_1_blue_1x+post_2_blue_1x)/2;
                sgdata.car1_y=(post_1_blue_1y+post_2_blue_1y)/2;
            }else if(post_1_blue_1x==0&&post_1_blue_1y==0&&post_2_blue_1x!=0&&post_2_blue_1y!=0){
                sgdata.car1=true;
                sgdata.car1_x=post_2_blue_1x;
                sgdata.car1_y=post_2_blue_1y;
            }else if(post_1_blue_1x!=0&&post_1_blue_1y!=0&&post_2_blue_1x==0&&post_2_blue_1y==0){
                sgdata.car1=true;
                sgdata.car1_x=post_1_blue_1x;
                sgdata.car1_y=post_1_blue_1y;
            }else 
            {
                sgdata.car1=false;
                
            }

            if(post_1_blue_2x!=0&&post_1_blue_2y!=0&&post_2_blue_2x!=0&&post_2_blue_2y!=0)
            {
                sgdata.car2=true;
                sgdata.car2_x=(post_1_blue_2x+post_2_blue_2x)/2;
                sgdata.car2_y=(post_1_blue_2y+post_2_blue_2y)/2;
            }else if(post_1_blue_2x==0&&post_1_blue_2y==0&&post_2_blue_2x!=0&&post_2_blue_2y!=0){
                sgdata.car2=true;
                sgdata.car2_x=post_2_blue_2x;
                sgdata.car2_y=post_2_blue_2y;
            }else if(post_1_blue_2x!=0&&post_1_blue_2y!=0&&post_2_blue_2x==0&&post_2_blue_2y==0){
                sgdata.car2=true;
                sgdata.car2_x=post_1_blue_2x;
                sgdata.car2_y=post_1_blue_2y;
            }else 
            {
                sgdata.car2=false;
                
            }

        
        
        
            if(post_1_red_1x!=0&&post_1_red_1y!=0&&post_2_red_1x!=0&&post_2_red_1y!=0)
            {
                sgdata.car3=true;
                sgdata.car3_x=(post_1_red_1x+post_2_red_1x)/2;
                sgdata.car3_y=(post_1_red_1y+post_2_red_1y)/2;
            }else if(post_1_red_1x==0&&post_1_red_1y==0&&post_2_red_1x!=0&&post_2_red_1y!=0){
                sgdata.car3=true;
                sgdata.car3_x=post_2_red_1x;
                sgdata.car3_y=post_2_red_1y;
            }else if(post_1_red_1x!=0&&post_1_red_1y!=0&&post_2_red_1x==0&&post_2_red_1y==0){
                sgdata.car3=true;
                sgdata.car3_x=post_1_red_1x;
                sgdata.car3_y=post_1_red_1y;
            }else 
            {
                sgdata.car3=false;
                
            }

            if(post_1_red_2x!=0&&post_1_red_2y!=0&&post_2_red_2x!=0&&post_2_red_2y!=0)
            {
                sgdata.car4=true;
                sgdata.car4_x=(post_1_red_2x+post_2_red_2x)/2;
                sgdata.car4_y=(post_1_red_2y+post_2_red_2y)/2;
            }else if(post_1_red_2x==0&&post_1_red_2y==0&&post_2_red_2x!=0&&post_2_red_2y!=0){
                sgdata.car4=true;
                sgdata.car4_x=post_2_red_2x;
                sgdata.car4_y=post_2_red_2y;
            }else if(post_1_red_2x!=0&&post_1_red_2y!=0&&post_2_red_2x==0&&post_2_red_2y==0){
                sgdata.car4=true;
                sgdata.car4_x=post_1_red_2x;
                sgdata.car4_y=post_1_red_2y;
            }else 
            {
                sgdata.car4=false;
                
            }
        
       // if(strstr(recv_buf,IP_FOUND))
                //{
                    cout<<"+++++++++++++++++++++++1+++++++++++++++++++++++++++++"<<endl;
                        //响应客户端请求
                        //strncpy(send_buf, IP_FOUND_ACK, strlen(IP_FOUND_ACK) + 1);
                        ret = sendto(sock_fd,(char*)& sgdata, sizeof(sgdata), 0, (struct sockaddr*)&recvaddr, socklen);//将数据发送给客户端
                        cout<<"****************************send to*********************"<<endl;
                        if(ret < 0 )
                        {
                                perror(" sendto! ");
                        }
                        printf(" send ack  msg to client !\n");
                //}


        //send_length = sendto(serverfd,(char*)& sgdata, sizeof(sgdata), 0, (struct sockaddr *) &clientaddr, client_addr_length);
        //cout<<"car1 "<<post_1_blue_1x<<" "<<post_1_blue_1y<<endl;
        
       // 离开多播组
        ret = setsockopt(sock_fd, IPPROTO_IP, IP_DROP_MEMBERSHIP, &mreq, sizeof(mreq));
        if(ret < 0){
                perror("IP_DROP_MEMBERSHIP");
                //freturn -1;
        }

        //close(sock_fd);//

        sleep(1);
    }
    
}

void zcx1()
{
    cout<<"************************zcx start********************************************"<<endl;
    if( (sock_fd1 = socket(AF_INET, SOCK_DGRAM, 0)) < 0 ){
         perror("socket() error");
         exit(1);
     }
    
    cout<<"+++++++++++++++++++++++2+++++++++++++++++++++++++++++"<<endl;

    socklen1 = sizeof(struct sockaddr);                

     // 通过struct sockaddr_in 结构设置服务器地址和监听端口；
     bzero(&localaddr1,sizeof(sock_fd1));
     //bzero(&clientaddr,sizeof(clientaddr));
     localaddr1.sin_family = AF_INET;
     localaddr1.sin_addr.s_addr = htonl(INADDR_ANY);
     localaddr1.sin_port = htons(6665);//htons(UDPPORT);
     local_addr_length1 = sizeof(localaddr1);

     // 使用bind() 函数绑定监听端口，将套接字文件描述符和地址类型变量（struct sockaddr_in ）进行绑定；
     if( bind(sock_fd1, (struct sockaddr *) &localaddr1, local_addr_length1) < 0){
         perror("bind() error");
         exit(1); 
     }
     cout<<"+++++++++++++++++++++++3+++++++++++++++++++++++++++++"<<endl;
    
     
       //设置多播的TTL值
        if(setsockopt(sock_fd1, IPPROTO_IP, IP_MULTICAST_TTL, &ttl1, sizeof(ttl1))<0){
                perror("IP_MULTICAST_TTL");
                //return -1;
        }
        //设置数据是否发送到本地回环接口
        if(setsockopt(sock_fd1, IPPROTO_IP, IP_MULTICAST_LOOP, &loop1, sizeof(loop1))<0){
                perror("IP_MULTICAST_LOOP");
                //return -1;
        }
        //加入多播组
        struct ip_mreq mreq1;
        mreq1.imr_multiaddr.s_addr=inet_addr(MCAST1);//多播组的IP
        mreq1.imr_interface.s_addr=htonl(INADDR_ANY);//本机的默认接口IP,本机的随机IP
        if(setsockopt(sock_fd1, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq1, sizeof(mreq1)) < 0){
                perror("IP_ADD_MEMBERSHIP");
                //return -1;
        }
     // 接收客户端的数据，使用recvfrom() 函数接收客户端的网络数据；
     recv_addr_length1 = sizeof(sockaddr_in);
    
    cout<<"+++++++++++++++++++++++4+++++++++++++++++++++++++++++"<<endl;
    
     int recv_length1 = 0;
     recv_length1 = recvfrom(sock_fd1, recv_buf1, sizeof(recv_buf1), 0, (struct sockaddr *) &recvaddr1, &socklen1);
     cout << "recv_length1 = "<< recv_length1 <<endl;
     cout << recv_buf1 << endl;

    cout<<"+++++++++++++++++++++++5+++++++++++++++++++++++++++++"<<endl;

     recv_addr_length1 = sizeof(sockaddr_in);
    while(1)
    {

        
        
            if(post_1_blue_1x!=0&&post_1_blue_1y!=0&&post_2_blue_1x!=0&&post_2_blue_1y!=0)
            {
                sgdata.car1=true;
                sgdata.car1_x=(post_1_blue_1x+post_2_blue_1x)/2;
                sgdata.car1_y=(post_1_blue_1y+post_2_blue_1y)/2;
            }else if(post_1_blue_1x==0&&post_1_blue_1y==0&&post_2_blue_1x!=0&&post_2_blue_1y!=0){
                sgdata.car1=true;
                sgdata.car1_x=post_2_blue_1x;
                sgdata.car1_y=post_2_blue_1y;
            }else if(post_1_blue_1x!=0&&post_1_blue_1y!=0&&post_2_blue_1x==0&&post_2_blue_1y==0){
                sgdata.car1=true;
                sgdata.car1_x=post_1_blue_1x;
                sgdata.car1_y=post_1_blue_1y;
            }else 
            {
                sgdata.car1=false;
                
            }

            if(post_1_blue_2x!=0&&post_1_blue_2y!=0&&post_2_blue_2x!=0&&post_2_blue_2y!=0)
            {
                sgdata.car2=true;
                sgdata.car2_x=(post_1_blue_2x+post_2_blue_2x)/2;
                sgdata.car2_y=(post_1_blue_2y+post_2_blue_2y)/2;
            }else if(post_1_blue_2x==0&&post_1_blue_2y==0&&post_2_blue_2x!=0&&post_2_blue_2y!=0){
                sgdata.car2=true;
                sgdata.car2_x=post_2_blue_2x;
                sgdata.car2_y=post_2_blue_2y;
            }else if(post_1_blue_2x!=0&&post_1_blue_2y!=0&&post_2_blue_2x==0&&post_2_blue_2y==0){
                sgdata.car2=true;
                sgdata.car2_x=post_1_blue_2x;
                sgdata.car2_y=post_1_blue_2y;
            }else 
            {
                sgdata.car2=false;
                
            }

        
        
        
            if(post_1_red_1x!=0&&post_1_red_1y!=0&&post_2_red_1x!=0&&post_2_red_1y!=0)
            {
                sgdata.car3=true;
                sgdata.car3_x=(post_1_red_1x+post_2_red_1x)/2;
                sgdata.car3_y=(post_1_red_1y+post_2_red_1y)/2;
            }else if(post_1_red_1x==0&&post_1_red_1y==0&&post_2_red_1x!=0&&post_2_red_1y!=0){
                sgdata.car3=true;
                sgdata.car3_x=post_2_red_1x;
                sgdata.car3_y=post_2_red_1y;
            }else if(post_1_red_1x!=0&&post_1_red_1y!=0&&post_2_red_1x==0&&post_2_red_1y==0){
                sgdata.car3=true;
                sgdata.car3_x=post_1_red_1x;
                sgdata.car3_y=post_1_red_1y;
            }else 
            {
                sgdata.car3=false;
                
            }

            if(post_1_red_2x!=0&&post_1_red_2y!=0&&post_2_red_2x!=0&&post_2_red_2y!=0)
            {
                sgdata.car4=true;
                sgdata.car4_x=(post_1_red_2x+post_2_red_2x)/2;
                sgdata.car4_y=(post_1_red_2y+post_2_red_2y)/2;
            }else if(post_1_red_2x==0&&post_1_red_2y==0&&post_2_red_2x!=0&&post_2_red_2y!=0){
                sgdata.car4=true;
                sgdata.car4_x=post_2_red_2x;
                sgdata.car4_y=post_2_red_2y;
            }else if(post_1_red_2x!=0&&post_1_red_2y!=0&&post_2_red_2x==0&&post_2_red_2y==0){
                sgdata.car4=true;
                sgdata.car4_x=post_1_red_2x;
                sgdata.car4_y=post_1_red_2y;
            }else 
            {
                sgdata.car4=false;
                
            }
        socklen = sizeof(struct sockaddr);
       // if(strstr(recv_buf,IP_FOUND))
                //{
                    cout<<"+++++++++++++++++++++++1+++++++++++++++++++++++++++++"<<endl;
                        //响应客户端请求
                        //strncpy(send_buf, IP_FOUND_ACK, strlen(IP_FOUND_ACK) + 1);
                        ret1 = sendto(sock_fd1,(char*)& sgdata, sizeof(sgdata), 0, (struct sockaddr*)&recvaddr1, socklen);//将数据发送给客户端
                        cout<<"****************************send to*********************"<<endl;
                        if(ret1 < 0 )
                        {
                                perror(" sendto! ");
                        }
                        printf(" send ack  msg to client !\n");
                //}


        //send_length = sendto(serverfd,(char*)& sgdata, sizeof(sgdata), 0, (struct sockaddr *) &clientaddr, client_addr_length);
        //cout<<"car1 "<<post_1_blue_1x<<" "<<post_1_blue_1y<<endl;
        
       // 离开多播组
        ret1 = setsockopt(sock_fd1, IPPROTO_IP, IP_DROP_MEMBERSHIP, &mreq1, sizeof(mreq1));
        if(ret1 < 0){
                perror("IP_DROP_MEMBERSHIP");
                //freturn -1;
        }

        //close(sock_fd);//

        sleep(1);
    }
    
}

int main(int argc, char** argv) {
	int pattren = 0;
	std::string engine_name = STR2(NET);
	engine_name = "yolov5" + engine_name + ".engine";
    
    if (argc == 2 && std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{ nullptr };
        APIToModel(BATCH_SIZE, &modelStream);
        assert(modelStream != nullptr);
        std::ofstream p(engine_name, std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 0;
    } else if (argc == 3 && std::string(argv[1]) == "-d") {
		pattren = 2;
	} 
	else {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./yolov5 -s  // serialize model to plan file" << std::endl;
        std::cerr << "./yolov5 -d ../samples  // deserialize plan file and run inference" << std::endl;
        return -1;
    }
    cam_initialize();
    //infergai(pattren, argv[2], engine_name);
    sgdata.car1=false;
    sgdata.car2=false;
    sgdata.car3=false;
    sgdata.car4=false;
    sgdata.car1_x=0;
    sgdata.car1_y=0;
    sgdata.car2_x=0;
    sgdata.car2_y=0;
    sgdata.car3_x=0;
    sgdata.car3_y=0;
    sgdata.car4_x=0;
    sgdata.car4_y=0;
    thread camera1(infergai,pattren,argv[2],engine_name);
    sleep(10);
    //thread camera2(infergai1,pattren,argv[2],engine_name);
    sleep(5);
    thread zcx_1(zcx);
    sleep(5);
    thread zcx_2(zcx1);
    sleep(1);
    camera1.join();
    //camera2.join();
    zcx_1.join();
    zcx_2.join();

    // Print histogram of the output distribution
    //std::cout << "\nOutput:\n\n";
    //for (unsigned int i = 0; i < OUTPUT_SIZE; i++)
    //{
    //    std::cout << prob[i] << ", ";
    //    if (i % 10 == 0) std::cout << std::endl;
    //}
    //std::cout << std::endl;

    return 0;
}
