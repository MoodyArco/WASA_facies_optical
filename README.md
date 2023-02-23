# WASA_facies_optical

This is the researsh after WASA_faciesXRF. The goal is shown as title, we want to use the photographs to learn the model, so we use 'optical' instead of XRF. In the early stage, I tried logisticregression (lr), randomforest (rf) and support vector classification (svc). The first try of these model is such a tragedy that the socre is 0.10 (lr), 0.13 (rf), 0.15(svc). The latest result of the model, we scale the RGB from 0-255 to 0-1, and we also increase the n_component to 150 in PCA parameter to get a beeter score: 0.18. 
