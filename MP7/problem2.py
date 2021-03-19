from klampt.io import loader
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor 
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from scipy.stats import randint,uniform
from PIL import Image
import os
import sys
import csv
import random
import pickle

PROBLEM = '2a'
#PROBLEM = '2b'
DEPTH_SCALE = 8000

grasp_attributes = ['score']
#3A: uncomment me to predict more features
#for attr in ['axis_heading','axis_elevation','opening']:

def load_images_dataset(folder):
    """Loads an image dataset from a folder.  Result is a list of
    (color,depth,camera_transform,grasp_attr) tuples.
    """
    metadatafn = os.path.join(folder,'metadata.csv')
    rows = []
    with open(metadatafn, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            rows.append(row)
    print("Read",len(rows)-1,"training images")
    if len(rows)<=1:
        raise RuntimeError("Hey, no rows read from metadata file?")
    cols = dict((v,i) for (i,v) in enumerate(rows[0]))
    dataset = []
    for i in range(1,len(rows)):
        color = np.asanyarray(Image.open(os.path.join(folder,rows[i][cols['color_fn']])))
        depth = np.asanyarray(Image.open(os.path.join(folder,rows[i][cols['depth_fn']])))*(1.0/DEPTH_SCALE)
        assert len(color.shape)==3 and color.shape[2]==3
        gripper_base_fn = os.path.join(folder,rows[i][cols['grasp_fn']])
        base,ext = os.path.splitext(gripper_base_fn)
        grasp_attrs = dict()
        for attr in grasp_attributes:
            grasp_channel = np.asanyarray(Image.open(base + '_' + attr + ext))*(1.0/255.0)
            grasp_attrs[attr] = grasp_channel
        dataset.append((color,depth,loader.read('RigidTransform',rows[i][cols['view_transform']]),grasp_attrs))
    return dataset

def get_region_of_interest(image,roi,fill_value='average'):
    """
    Retrieves a subset of an image, being friendly to image boundaries.

    Args
        image (np.ndarray): has at least 2 dimensions
        roi (tuple): values (i1,i2,j1,j2) defining the patch boundaries
            [i1:i2,j1:j2] (note: non-inclusive of i2 and j2).
        fillvalue: a value to fill in when the roi expands beyond the
            boundaries. Can also be 'average'
    """
    i1,i2,j1,j2 = roi
    if i1 < 0 or i2 > image.shape[0] or j1 < 0 or j2 > image.shape[1]:
        subset = image[max(i1,0):min(i2,image.shape[0]),
                    max(j1,0):min(j2,image.shape[1])]
        paddings = [(max(-i1,0),max(i2-image.shape[0],0)),
                    (max(-j1,0),max(j2-image.shape[1],0))]
        if len(image.shape) > 2:
            paddings += [(0,0)]*(len(image.shape)-2)
        if fill_value == 'average':
            if len(subset)==0:
                fill_value = 0
            else:
                fill_value = np.average(subset)
        res = np.pad(subset,tuple(paddings),mode='constant',constant_values=(fill_value,))
        assert res.shape[0] == i2-i1,"Uh... mismatch? {} vs {} (roi {})".format(res.shape[0],i2-i1,roi)
        assert res.shape[1] == j2-j1,"Uh... mismatch? {} vs {} (roi {})".format(res.shape[1],j2-j1,roi)
        return res
    else:
        return image[i1:i2,j1:j2]

def set_region_of_interest(image,roi,value):
    """Sets a patch of an image to some value, being tolerant to the
    boundaries of the image.
    """
    i1,i2,j1,j2 = roi
    image[max(i1,0):min(i2,image.shape[0]),max(j1,0):min(j2,image.shape[1])] = value


def make_patch_dataset(dataset,predicted_attr='score',patch_size=30):
    """Create a matrix (X,y) consisting of flattened feature vectors
    from the image dataset.
    """
    #TODO: tune me / fill me in for Problem 2a
    samples_per_image = 100
    patch_radius = patch_size//2
    A = []
    b = []
    for image in dataset:
        color,depth,transform,grasp_attrs = image
        #you might want these?
        color_gradient_x = np.linalg.norm(color[1:,:,:]-color[:-1,:,:],axis=2)
        color_gradient_y = np.linalg.norm(color[:,1:,:]-color[:,:-1,:],axis=2)
        depth_gradient_x = depth[1:,:]-depth[:-1,:]
        depth_gradient_y = depth[:,1:]-depth[:,:-1]
        output = grasp_attrs[predicted_attr]
        scores = []
        for i in range(samples_per_image):
            x,y = random.randint(patch_radius,color.shape[1]-1-patch_radius),random.randint(patch_radius,color.shape[0]-1-patch_radius)
            
            roi = (y-patch_radius,y+patch_radius,x-patch_radius,x+patch_radius)
            patch1 = get_region_of_interest(color,roi).flatten()
            patch2 = get_region_of_interest(depth,roi).flatten()
            A.append(np.hstack((patch1,patch2)))
            assert len(A[-1].shape)==1
            assert A[-1].shape == A[0].shape
            b.append(output[y,x])
    return np.vstack(A),np.array(b)

def train_predictor(X,y):
    #TODO: tune me for problem 2a
    print("Training on dataset with",X.shape[0],"observations with dimension",X.shape[1])
    print("Average score",np.average(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    #Select your learning model / pipeline here
    estimators = [('pca', PCA(n_components=20,whiten=True)), ('linear', LinearRegression())]
    # estimators = [('pca', PCA(n_components=20,whiten=True)), ('rf', RandomForestRegressor(random_state=0,n_estimators=10,max_depth=10))]
    # estimators = [('pca', PCA(n_components=20,whiten=True)), ('gb',GradientBoostingRegressor(random_state=0,n_estimators=100,learning_rate=0.1))]
    # estimators = [('pca',PCA(n_components=20,whiten=True)), ('mlp',MLPRegressor(hidden_layer_sizes=(100,)))]

    pipe = Pipeline(estimators)

    pipe.fit(X_train, y_train)
    model = pipe

    # # to do model selection, create a searchCV object and fit it to the data
    # # define the parameter space that will be searched over
    # param_distributions = {'pca__n_components': randint(5,50)}
    # param_distributions = {'pca__n_components': randint(5,50),
    #                         'rf__n_estimators': randint(1, 5),
    #                         'rf__max_depth': randint(5, 10) }
    # param_distributions = {'pca__n_components': randint(5,50),
    #                         'gb__n_estimators': randint(1, 5),
    #                         'gb__learning_rate': uniform(loc=0.3, scale=0.29)}
    # search = RandomizedSearchCV(estimator=pipe,
    #                             n_iter=5,
    #                             param_distributions=param_distributions,
    #                             random_state=0)
    # search.fit(X_train, y_train)
    # print(search.best_params_)
    # model = search
    # the search object now acts like a normal random forest estimator

    print("Test score",model.score(X_test, y_test))
    print("Constant predictor RMSE",np.linalg.norm(y_test-np.average(y_train))/np.sqrt(len(y_test)))
    print("Train RMSE",np.linalg.norm(model.predict(X_train)-y_train)/np.sqrt(len(y_train)))
    print("Test RMSE",np.linalg.norm(model.predict(X_test)-y_test)/np.sqrt(len(y_test)))
    return model

def predict_patches(image_group,pts,model,patch_size=30):
    """Returns predictions of the given model for the given points in
    the image.  Make sure you are calculating features of these
    points exactly like you did in make_patch_dataset.
    """
    #TODO: fill me in for problem 2B
    patch_radius = patch_size//2
    color,depth,transform,grasp_attrs = image_group
    preds = []
    for i,(x,y) in enumerate(pts):
        if len(pts) > 10000 and i%(len(pts)//10)==0:
            print(i//(len(pts)//10)*10,"...")
        roi = (y-patch_radius,y+patch_radius,x-patch_radius,x+patch_radius)
        patch1 = get_region_of_interest(color,roi).flatten()
        patch2 = get_region_of_interest(depth,roi).flatten()
        preds.append(model.predict([np.hstack((patch1,patch2))])[0])
    return preds

def gen_prediction_images(attr='score'):
    """Creates prediction images and saves them to predictions/
    using your predict_patches function.
    """
    import pickle
    with open('trained_model_{}.pkl'.format(attr),'rb') as f:
        model = pickle.load(f)
    dataset = load_images_dataset('image_dataset')
    h,w = dataset[0][1].shape
    patch_radius = 30//2
    X = list(range(patch_radius,w-patch_radius))
    Y = list(range(patch_radius,h-patch_radius))
    pts = np.transpose([np.tile(X, len(Y)), np.repeat(Y, len(X))])
    try:
        os.mkdir('predictions')
    except Exception:
        pass
    img_skip = 10
    for i,image in enumerate(dataset[::img_skip]):
        print("Predicting",len(pts),"for image",i)
        values = predict_patches(image,pts,model)
        pred = np.zeros((h,w))
        for pt,v in zip(pts,values):
            x,y = pt
            pred[y,x] = v
        pred_quantized = (pred*255.0).astype(np.uint8)
        filename = "predictions/image_%04d.png"%(i,)
        Image.fromarray(pred_quantized).save(filename)

def gen_partitioned_image_features(image_group,patch_size,roi=None):
    """Given a training set instance and a patch size, generates a
    pair (X,patches) where patches are non-overlapping ROIs of size
    patch_size, and X is a list of image feature vectors.
    """
    #TODO: implement me with the same features you selected for problem 2a
    color,depth,transform,grasp_attrs = image_group
    iofs,jofs = 0,0
    if roi is not None:
        #a ROI was specified -- sample patches only from that region
        i1,i2,j1,j2 = roi
        iofs,jofs = i1,j1
        color = get_region_of_interest(color,roi)
        depth = get_region_of_interest(depth,roi)
    h,w = color.shape[:2]
    A = []
    patches = []
    #TODO: Problem 2C: iterate over patches
    roi = (0,patch_size,0,patch_size)
    patch1 = get_region_of_interest(color,roi).flatten()
    patch2 = get_region_of_interest(depth,roi).flatten()
    A.append(np.hstack((patch1,patch2)))
    #if an ROI was selected, don't forget to translate back to full image coordinates
    i1,i2,j1,j2 = roi
    patches.append((i1+iofs,i2+iofs,j1+jofs,j2+jofs))
    return A,patches

def gen_partitioned_dataset(dataset,patch_size):
    """For faster predictions, this generates a dataset
    (X,y) of patches and maximum scores over the patches.
    """
    predicted_attr = 'score'
    A = []
    b = []
    for image in dataset:
        color,depth,transform,grasp_attrs = image
        output = grasp_attrs[predicted_attr]
        features, patches = gen_partitioned_image_features(image,patch_size)
        A += features
        #predict output = the best score in the patch
        for roi in patches:
            b.append(get_region_of_interest(output,roi,0).max())
    return np.vstack(A),np.array(b)

def train_faster_prediction():
    """Learns two extra models: 1) mapping 80x80 patches to the
    predicted max score over the patch, and 2) mapping 20x20
    patches to the predicted max score over the patch.
    """
    dataset = load_images_dataset('image_dataset')
    X80,y80=gen_partitioned_dataset(dataset,80)
    model = train_predictor(X80,y80)
    with open('trained_model_80x80_maxscore.pkl','wb') as f:
        pickle.dump(model,f)
    X20,y20=gen_partitioned_dataset(dataset,20)
    model = train_predictor(X20,y20)
    with open('trained_model_20x20_maxscore.pkl','wb') as f:
        pickle.dump(model,f)

def predict_scores_faster(image,models):
    """Returns predicted scores across the entire image group"""
    #TODO: complete me for problem 2D.
    model_80, model_20, model_pixel = models
    pred_image = np.zeros(image[0].shape[:2])
    features,patches = gen_partitioned_image_features(image,80)
    print("Predicting for",len(features),"80x80 patches")
    preds = model_80.predict(np.vstack(features))
    for patch,pred in zip(patches,preds):
        #fill in the predicted patches with their max scores
        set_region_of_interest(pred_image,patch,min(1,max(0,pred*0.25)))
    return pred_image


def gen_faster_prediction_images():
    """Generates prediction images using the faster multiresolution
    approach. 
    """
    with open('trained_model_80x80_maxscore.pkl','rb') as f:
        model_80 = pickle.load(f)
    with open('trained_model_20x20_maxscore.pkl','rb') as f:
        model_20 = pickle.load(f)
    with open('trained_model_score.pkl','rb') as f:
        model_pixel = pickle.load(f)
    
    dataset = load_images_dataset('image_dataset')
    h,w = dataset[0][1].shape
    
    try:
        os.mkdir('predictions')
    except Exception:
        pass
    img_skip = 10
    for img_index,image in enumerate(dataset[::img_skip]):
        print("Predicting for image",img_index)
        pred_image = predict_scores_faster(image,[model_80,model_20,model_pixel])
        pred_quantized = (pred_image*255.0).astype(np.uint8)
        filename = "predictions/image_%04d_fast.png"%(img_index,)
        Image.fromarray(pred_quantized).save(filename)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        PROBLEM = sys.argv[1]
    if PROBLEM == '2a':
        dataset = load_images_dataset('image_dataset')
        
        for attr in grasp_attributes:
            X,y = make_patch_dataset(dataset,attr)
            model = train_predictor(X,y)
            with open('trained_model_{}.pkl'.format(attr),'wb') as f:
                pickle.dump(model,f)
    elif PROBLEM == '2b':
        gen_prediction_images()
    elif PROBLEM == '2c':
        train_faster_prediction()
    elif PROBLEM == '2d':
        gen_faster_prediction_images()
    else:
        raise ValueError("Invalid PROBLEM?")