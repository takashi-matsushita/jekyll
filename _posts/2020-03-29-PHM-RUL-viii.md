---
layout: post
title:  Turbofan Engine Degradation Simulation Data Set - VIII
author: Takashi MATSUSHITA
categories: AI
tags: [Regression,]
---
### Complex models with denoising

```text
      Linear    ElasticNet  SVR     xgboost
==========================================================================
raw
---------------------------------------------------------------------------
MSE   1197.8    1167.3       736.0   840.3
R2    0.6818    0.6899      0.8045  0.7768
score inf       inf         inf     inf

ma/fe
---------------------------------------------------------------------------
MSE   1057.7     959.1       966.6   610.3
R2    0.6950    0.7234      0.7212  0.8240
score inf       inf         inf     inf
```

The fit results of the complex model have also been improved by denoising sensor readings with moving average except for SVR. Actually, for SVR and xgboost, hyper-parameter tuning to be performed to get optimal model. The results above have been obtained with the default parameter settings for SVR and xgboost models.
