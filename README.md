# FIT-Net
This repository is the official implementation of **FIT-Net: Feature Interaction Transformer Network for Pathologic Myopia Diagnosis.** [FIT-Net](https://ieeexplore.ieee.org/abstract/document/10087215)   
If you use the codes and models from this repo, please cite our work. Thanks!  
``` 
@ARTICLE{10087215,
  author={Chen, Shaobin and Wu, Zhenquan and Li, Mingzhu and Zhu, Yun and Xie, Hai and Yang, Peng and Zhao, Cheng and Zhang, Yongtao and Zhang, Shaochong and Zhao, Xinyu and Wang, Tianfu and Lu, Lin and Zhang, Guoming and Lei, Baiying},
  journal={IEEE Transactions on Medical Imaging}, 
  title={FIT-Net: Feature Interaction Transformer Network for Pathologic Myopia Diagnosis}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TMI.2023.3260990}}    
```   
## Data Preparation
**Divide the data into k-folds for cross-validation**.  
If k-fold cross-validation is used during training, you can use `create_excelFile_for_k_fold_dataset.py` to split the data into k-folds.  
Before using this script, make sure that your image data is stored in different folders by category, as shown below:  
![dataFolder](https://raw.githubusercontent.com/chenshaobin/FITNet/master/images/dataFolder.jpg)  
```python

excelFileSavePath = r'xxx\xxx.xlsx'  
imgRootPath = r'xxx\imgPath'  
kFold = 10  
mainObj = get_k_fold_Data(excelFileSavePath=excelFileSavePath, imgRootPath=imgRootPath, kFold=kFold, num_class=4)  
mainObj.getKFoldData()  
```  

After that, the name of the picture and the corresponding label will be stored in the excel file：   
![data_excel_format](https://raw.githubusercontent.com/chenshaobin/FITNet/master/images/data_excel_format.jpg)   
   