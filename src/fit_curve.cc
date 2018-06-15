#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include <iostream>
#include <utility>
#include <vector>

int main(int argc, char* argv[]) {
    // Initialize a tensorflow session
    tensorflow::Session *session;
    tensorflow::Status status = tensorflow::NewSession(tensorflow::SessionOptions(), &session);
    if (!status.ok()) {
        std::cout << status.ToString() << std::endl;
        return 1;
    }

    // Read in the protobuf graph we exported
    // (The path seems to be relative to the cwd. Keep this in mind
    // when using `bazel run` since the cwd isn't where you call
    // `bazel run` but from inside a temp folder.)
    std::cout << "Loading graph" << std::endl;
    tensorflow::GraphDef graph_def;
    status = tensorflow::ReadBinaryProto(tensorflow::Env::Default(), "/home/dalmo/work/tensors/models/fit_curve.pb", &graph_def);
    if (!status.ok()) {
        std::cout << status.ToString() << std::endl;
        return 1;
    }

    // Add the graph to the session
    status = session->Create(graph_def);
    if (!status.ok()) {
        std::cout << status.ToString() << std::endl;
        return 1;
    }

    // Setup inputs and outputs:

    // Our graph doesn't require any inputs, since it specifies default values,
    // but we'll change an input to demonstrate.
    std::cout << "Creating X tensor" << std::endl;
    tensorflow::Tensor X(tensorflow::DT_FLOAT, tensorflow::TensorShape({8, 1}));
    auto xMatrix = X.matrix<float>();
    xMatrix(0) = 6.83;
    xMatrix(1) = 4.668;
    xMatrix(2) = 8.9;
    xMatrix(3) = 7.91;
    xMatrix(4) = 5.7;
    xMatrix(5) = 8.7;
    xMatrix(6) = 3.1;
    xMatrix(7) = 2.1;

    std::cout << "Creating Y tensor" << std::endl;
    tensorflow::Tensor Y(tensorflow::DT_FLOAT, tensorflow::TensorShape({8, 1}));
    auto yMatrix = Y.matrix<float>();
    yMatrix(0) = 1.84;
    yMatrix(1) = 2.273;
    yMatrix(2) = 3.2;
    yMatrix(3) = 2.831;
    yMatrix(4) = 2.92;
    yMatrix(5) = 3.24;
    yMatrix(6) = 1.35;
    yMatrix(7) = 1.03;

    for (int i = 0; i < 8; ++i) {
        std::cout << "xMatrix(" << i << "): " << xMatrix(i) <<
                     ", yMatrix(" << i << "): " << yMatrix(i) << std::endl;
    }

    tensorflow::Tensor W(tensorflow::DT_FLOAT, tensorflow::TensorShape());
    W.scalar<float>()() = 3.0;

    tensorflow::Tensor b(tensorflow::DT_FLOAT, tensorflow::TensorShape());
    b.scalar<float>()() = 2.0;

    std::cout << "Populating inputs" << std::endl;
    std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
        {"X", X},
        {"Y", Y},
        {"W", W},
        {"b", b},
    };

    // The session will initialize the outputs
    std::vector<tensorflow::Tensor> outputs;

    // Run the session, evaluating our "c" operation from the graph
    std::cout << "Will run" << std::endl;
    status = session->Run(inputs, {"prediction"}, {}, &outputs);
    if (!status.ok()) {
        std::cout << "Error predicting" << status.ToString() << std::endl;
        return 1;
    }
    std::cout << "Did run" << std::endl;

    std::cout << "Outputs size: " << outputs.size() << std::endl;

    // Grab the first output (we only evaluated one graph node: "c")
    // and convert the node to a scalar representation.
    // for (const auto &output : outputs) {
    //     std::cout << "output: " << output.scalar<float>() << std::endl;
    // }
    // auto output_c = outputs[0].matrix<float>()(0);
    auto output_c = outputs[0].scalar<float>();
    // auto output_c = outputs[0].vector<float>()(0);

    // (There are similar methods for vectors and matrices here:
    // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/public/tensor.h)

    // Print the results
    // std::cout << outputs[0].DebugString() << std::endl;
    std::cout << output_c << std::endl;

    // Free any resources used by the session
    session->Close();
    return 0;
}
