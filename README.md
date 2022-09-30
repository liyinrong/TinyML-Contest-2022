# Information about files in the repo

The environment required is the same as that of demo training code.

To reduce the package size, the training dataset is not included. Please copy the complete set into `./tinyml_contest_data_training` first.

To start the training process, please run

    python training.py

Several hyper-parameters may affect the training results, please check them at the end of `training.py`.

The training script saves the model to local disk every several training cycles. You can see them in `./saved_models` together with the related test results.

To transform the trained model into onnx format so that it could be deployed on evaluation board with CubeMX, please rename the selected model with `IEGM_net.pkl` and put it in `./saved_models`, then run

    python pkl2onnx.py

Authors: Siyuan, Shaoqiang, Yuedong, Yinrong