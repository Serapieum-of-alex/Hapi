# Muskingum 

Muskingum is a hydrologic-routing method which employs the equation of continuity to predict magnitude, volume and temporal patterns of flow as it translates downstream of a channel.

```
𝐼−𝑄 = 𝑑𝑆/𝑑𝑡
```

![HBV Component](../img/muskingum1.png) ![HBV Component](../img/muskingum2.png)

channel routing functions of inflow, outflow and storage where storage can be considered as two parts, prism & wedge storage.

```
S=K\ast\left[x\ast I^m+\left(1-x\right)\ast Q^m\right]
```