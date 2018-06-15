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
    tensorflow::GraphDef graph_def;
    status = tensorflow::ReadBinaryProto(tensorflow::Env::Default(), "/home/dalmo/work/tensors/models/graph.pb", &graph_def);
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
    tensorflow::Tensor t1(tensorflow::DT_FLOAT, tensorflow::TensorShape());
    t1.scalar<float>()() = 3.0;

    tensorflow::Tensor b(tensorflow::DT_FLOAT, tensorflow::TensorShape());
    b.scalar<float>()() = 2.0;

    std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
        { "a", t1 },
        { "b", b },
    };

    // The session will initialize the outputs
    std::vector<tensorflow::Tensor> outputs;

    // Run the session, evaluating our "c" operation from the graph
    status = session->Run(inputs, {"mult"}, {}, &outputs);
    if (!status.ok()) {
        std::cout << status.ToString() << std::endl;
        return 1;
    }

    // Grab the first output (we only evaluated one graph node: "c")
    // and convert the node to a scalar representation.
    auto output_c = outputs[0].scalar<float>();

    // (There are similar methods for vectors and matrices here:
    // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/public/tensor.h)

    // Print the results
    std::cout << outputs[0].DebugString() << std::endl;
    std::cout << output_c() << std::endl;

    // Free any resources used by the session
    session->Close();
    return 0;
}

