load(
    "@org_tensorflow//tensorflow:tensorflow.bzl",
    "tf_cc_binary",
)

tf_cc_binary(
    name = "tensors",
    srcs = ["main.cpp"],
    deps = [
        "@org_tensorflow//tensorflow/cc:cc_ops",
        "@org_tensorflow//tensorflow/cc:client_session",
        "@org_tensorflow//tensorflow/core:tensorflow",
    ],
)

tf_cc_binary(
    name = "fit_curve",
    srcs = ["fit_curve.cc"],
    deps = [
        "@org_tensorflow//tensorflow/cc:cc_ops",
        "@org_tensorflow//tensorflow/cc:client_session",
        "@org_tensorflow//tensorflow/core:tensorflow",
    ],
)
