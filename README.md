### Run NCC Match & LSM Match：
```
1.Extract Match.zip to Match folder, download opencv4.5.4 to replace opencv4.5.4 folder in Match.
2.To run the program, please build the whole project sln based on Match.cpp and configure opencv c++(include and lib, dll) first.
Note: Match.cpp is the file that contains the main function. Because opencv4.5.4 and related compiled files are too large, so only the cpp files are submitted.
```
### Run Sift
```
1.Unzip Sift.zip to Sift file
2.Please configure opencv, numpy, matplotlib, time(pip install) in the virtual environment
3. Execute python main.py in the command line to run feature matching and feature extraction
Note: Due to the lack of CUDA acceleration, the operation speed is very slow and takes about ten minutes, please be patient and wait for the result.If you need to see the intermediate results such as feature extraction results, you can remove the comments of cv2.imshow.
```
