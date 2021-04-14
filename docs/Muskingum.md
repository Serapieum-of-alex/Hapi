# Muskingum 

Muskingum is a hydrologic-routing method which employs the equation of continuity to predict magnitude, volume and temporal patterns of flow as it translates downstream of a channel.

```
𝐼−𝑄 = 𝑑𝑆/𝑑𝑡
```

![HBV Component](../img/muskingum1.png) ![HBV Component](../img/muskingum2.png)

Channel routing functions of inflow, outflow and storage where storage can be considered as two parts, prism & wedge storage.


<img src="https://latex.codecogs.com/svg.latex?\fn_jvn&space;S&space;=&space;K\ast\left[x\ast&space;I^m&plus;\left(1-x\right)\ast&space;Q^m\right]" title="S = K\ast\left[x\ast I^m+\left(1-x\right)\ast Q^m\right]" />

```
Where k is the travel time constant and x are weighting coefficient to determine the linearity of the water surface, and it ranges between 0 & 0.5, and m is an exponential constant varies from 0.6 for rectangle channel to 1.
```

For Muskingum version of the channel routing equation m equals one which made the relation between S and I, Q. Using coefficient k & x three weights can be calculated as follow:

Coefficient of Muskingum equation

<img src="https://latex.codecogs.com/svg.latex?\fn_jvn&space;\large&space;C1=\frac{\mathrm{\Delta&space;t}-2KX}{2K\left(1-X\right)&plus;\mathrm{\Delta&space;t}},\&space;C2=\frac{\mathrm{\Delta&space;t}&plus;2KX}{2K\left(1-X\right)&plus;\mathrm{\Delta&space;t}}\&space;C3=\frac{2K\left(1-X\right)-\mathrm{\Delta&space;t}}{2K\left(1-X\right)&plus;\mathrm{\Delta&space;t}}" title="\large C1=\frac{\mathrm{\Delta t}-2KX}{2K\left(1-X\right)+\mathrm{\Delta t}},\ C2=\frac{\mathrm{\Delta t}+2KX}{2K\left(1-X\right)+\mathrm{\Delta t}}\ C3=\frac{2K\left(1-X\right)-\mathrm{\Delta t}}{2K\left(1-X\right)+\mathrm{\Delta t}}" />

To route the inflow hydrograph

Muskingum equation

<img src="https://latex.codecogs.com/gif.latex?\large&space;Q&space;=&space;C1&space;*&space;I_{j&plus;1}&space;&plus;&space;C2&space;*&space;I_j&space;&plus;&space;C3&space;*&space;Q_j" title="\large Q = C1 * I_{j+1} + C2 * I_j + C3 * Q_j" />
