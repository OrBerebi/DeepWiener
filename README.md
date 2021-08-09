## DeepWiener
Signal enhancement of spatial recording using a Wiener space-time-frequency mask, estimated by a UNet based deep neural network.

#How to run it
1. Open ./matlab/scripts/get_mesurments.m using Matlab 2020b (I think it will work on earlier versions as well)
  1.1. Here you can generate the dataset for the project, set the current_folder path.
  1.2. You need to set the dry_speakers_train_path, dry_speakers_test_path, dry_noise_path. This files are not avalable from this Github, due to large size.
  1.3. set the data_type flag (only one can be true)
  1.4. set the data size

