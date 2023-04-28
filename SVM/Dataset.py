import numpy as np
from sklearn.datasets import dump_svmlight_file
import os
import copy
from tqdm import tqdm
from libsvm.svmutil import svm_read_problem

class LIBSVM_Dataset():
    
    def __init__(self, generator=None) -> None:
        """
        ## parameter:
        - generator:
        A function, have to return 
            a dict ```{'data':X,'label':Y}```
                - X, Y are np.ndarray
        
            if it is None, it means no need to generate and just
            using the exist file directly
        
        """
        self.__generator = generator
        self.__data:dict = {}
        self.__libsvmdata:dict = {}
        
    def gen(self, k_fold=0, savepath=os.path.join("data"), **kwargs)->dict:
        
        """
        ## parameters:
            - generator: A function, have to return 
            a dict ```{'data':X,'label':Y}```
                - X, Y are np.ndarray
            
            please note that if self.__generator is None, 
            Calling this member function is illegal.
            
            - generator_para: A dict that 
            contains the parameters ```generator``` needs.
            
            - k_fold : split the generated data from
            ```generator``` to do k-fold cross validation. 
                - if it's 0, it will not split data 
                
            - savepath : will use dump_svmlight_file 
            from ```sklearn.dataset``` to make the data
            for libsvm needed format. Default is ./data/
                - if k_fold is not 0, will make ./datasavepath/0/,
                ... ./datasavepath/k_fold-1/ to save 
                the splited data.
                
                
        Once the data is generated, since it is np.ndarray,
        it will direclty be saved to ```savepath``` in order to 
        get libsvm data format by using svm_read_problem from ```libsvm.svmutil```
        """
        
        if self.__generator is None:
            print("Error ! no generator is given")
            return None
        
        self.__data['whole'] = self.__generator(**kwargs)
        
        if k_fold:
            self.__data['cv'] = self.k_foldcv_split_data(
                data=self.__data, k=k_fold
            )
        
        return self.get_data(fromfile=self.savedata(savepath=savepath))
                
    def k_foldcv_split_data(self, data:dict, k=5)->list: 
        
        ret = []
        N = data['whole']['data'].shape[0]
        indiecs = np.random.permutation(np.arange(N))
        foldsize = N//k
        folds = []
        if N%k == 0 :
            # can split to k-folds equally
            for i in range(k):
                b = i*foldsize
                e = b+foldsize
                folds.append([
                    data['whole']['data'][indiecs[b:e]],
                    data['whole']['label'][indiecs[b:e]]
                ])
            
        for i, testd in enumerate(folds):
            training = {} 
            training['data'] = np.vstack(
                list(_[0] for j, _ in enumerate(folds) if j != i)
            )
            training['label'] = np.vstack(
                list(_[1] for j, _ in enumerate(folds) if j!= i)
            )
            ret.append(
                {
                    'training':training,
                    'testing':{
                        'data':testd[0],
                        'label':testd[1]
                    }
                }
            )
        
        return ret
  
    def __walk_dir(self, root)->dict:
        files = {'whole':None, 'cv':[]}
        cvidx = 0
        for r, ds, fs in os.walk(root):
            if r == root:
                files['whole']=os.path.join(r,fs[0])
            else:
                files['cv'].append({})
                for f in fs:
                    if f == "train.svm":
                        files['cv'][cvidx]['training'] = os.path.join(r, f)
                    elif f == "test.svm":
                        files['cv'][cvidx]['testing'] = os.path.join(r, f)
                cvidx += 1
        
        return files

    def get_data(self, fromfile=None):
    
        if fromfile is None:
            if self.__libsvmdata is not None:
                return self.__libsvmdata
            else:
                print("Error !")
                print("Since no data has been generated, it needs files dir to load data")
                print("will return None")
                return None
        files = fromfile
        
        if isinstance(files, str):
            files = self.__walk_dir(fromfile)
        
        ret = {}
        for d, fname in files.items():
            if d == "whole":
                y, x = svm_read_problem(data_file_name=fname)
                ret[d] = {'data':x,'label':y}
            elif d == "cv":
                ret[d] = []
                for di in fname:
                    trainy, trainx = svm_read_problem(di['training'])
                    testy, testx = svm_read_problem(di['testing']) 
                    ret[d].append(
                        {
                            'training':{'data':trainx, 'label':trainy},
                            'testing':{'data':testx, 'label':testy}
                        }
                    )                
            
        self.__libsvmdata = copy.deepcopy(ret)
        return ret
       
    def savedata(self, savepath)->dict:
        """
        will return the hierarchy path =
        ```
        {
            'whole':whole_data_file_path,
            'cv':[
                {
                    'training':training_i_savepath,
                    'testing':testing_i_savepath
                }, ...
            ]
        }
        ```
        """
        
        ret = {}
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        
        svmdata = os.path.join(savepath, "whole.svm")
        print(f"writting whole data to {svmdata} ..")
        
        with open(svmdata, "wb+") as f:
            dump_svmlight_file(
                X=self.__data['whole']['data'], 
                y=self.__data['whole']['label'].reshape(-1,), 
                f=f
            )
        ret['whole'] = svmdata

        if 'cv' in self.__data:
            ret['cv'] = []
            print("writting cross validation data ..")
            cvdata_pbar = tqdm(self.__data['cv'])
            for _ ,cvi in enumerate(cvdata_pbar):
                
                cvi_savepath =os.path.join(savepath,f'{_}')
                if not os.path.exists(cvi_savepath):
                    os.mkdir(cvi_savepath)
                    
                cvdata_pbar.set_postfix_str(cvi_savepath)
                svmdata_traini = os.path.join(cvi_savepath, "train.svm")
                svmdata_testi = os.path.join(cvi_savepath, "test.svm")
                
                with open(svmdata_traini, "wb+") as f:
                    dump_svmlight_file(
                        X=cvi['training']['data'], 
                        y=cvi['training']['label'].reshape(-1,), 
                        f=f
                    )
                
                with open(svmdata_testi,"wb+") as f:
                    dump_svmlight_file(
                        X=cvi['testing']['data'], 
                        y=cvi['testing']['label'].reshape(-1,),
                        f=f
                    )
                ret['cv'].append(
                    {'training':svmdata_traini,'testing':svmdata_testi}
                )
        
        return ret
