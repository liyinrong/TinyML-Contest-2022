import zipfile

target_path = './'
zip_file = zipfile.ZipFile('tinyml_contest_data_training.zip')
zip_file.extractall(target_path)