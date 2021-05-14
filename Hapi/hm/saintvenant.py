# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 22:25:59 2021

@author: mofarrag
"""
import numpy as np
CWEIR = 1.704
BWIDTH = 10
HLIN = 0.001
power = 2/3

class SaintVenant():

    def __init__():
        pass

    def HBC(Sub, River, Hbnd, dt, dx,inih,storewl, MinQ):

        xs = np.zeros(shape=(Sub.XSno,8))
        Diff_coeff = np.zeros(shape=(Sub.XSno,3))
        XSarea = np.zeros(shape=(Sub.XSno))

        wl = np.zeros(shape=(Sub.XSno))
        hx = np.zeros(shape=(Sub.XSno))
        qx = np.zeros(shape=(Sub.XSno))

        h = np.zeros(shape=(Sub.XSno,River.TS))
        q = np.zeros(shape=(Sub.XSno,River.TS))

        storewl = np.zeros(shape=(River.XSno,2))
        # OverTopFlow = np.zeros(shape=(River.XSno*2,24))
        # OverTopWL = np.zeros(shape=(River.XSno*2,24))
        Lateral_q = MinQ.values[:-1,:-2]


        q_outL = np.zeros(shape=(Sub.XSno,River.TS))
        q_outR = np.zeros(shape=(Sub.XSno,River.TS))
        q_out = np.zeros(shape=(Sub.XSno,River.TS))

        q_outLx = np.zeros(shape=(Sub.XSno))
        q_outRx = np.zeros(shape=(Sub.XSno))
        q_outx = np.zeros(shape=(Sub.XSno))


        for t in range(0,River.TS):
            dtrest = dt
             # IC
            if t == 0:
                # fill the water depth array with the initial condition at time step 1
                hx[:] = inih
            else :
                #water level in the previous time-step is now turned to be the current
                hx = h[:,t-1]

             # boundary condition
             # water level at the first cross section (x=1]is the BC
            hx[0] = Hbnd[0]
             # hx is the water depth of the previous time step
             # except for the first cross section water depth is the boundary condition
             # for this time step
             # hx=[t,t-1,t-1,t-1,.............]
            adaptiveTS_counter = 0
            # for each time step use adaptive time step
            # adaptive time step -----------------------------------------------------
            while dtrest > 0.001:
                adaptiveTS_counter = adaptiveTS_counter + 1
                # calculate the area and perimeter based on the water depth of the previous time step
                # for the whole cross sections in the sub-basin
                for x in range(0, Sub.XSno):
                    # xs[:,:] = 0
                    # calculate the area & perimeter of the whole XS
                    Coords = River.GetVortices(hx[x], Sub.hl[x], Sub.hr[x], Sub.cl[x], Sub.cr[x], Sub.mw[x],Sub.Dbf[x])
                    xs[x,0:2] = River.PolygonGeometry(Coords)

                    # area & perimeter of the Upper part only
                    if hx[x] <= Sub.Dbf[x]:
                        xs[x,2] = 0
                        xs[x,3] = 0
                    else :
                    # area of upper part only
                        xs[x,2] =  xs[x,0] - (Sub.Dbf[x] * Sub.mw[x])
                    # perimeter of upper part
                        xs[x,3] =  xs[x,1] -  (2* Sub.Dbf[x] + Sub.mw[x])

                    # area & perimeter of the lower part only
                    xs[x,4] = xs[x,0] - xs[x,2]
                    xs[x,5] = xs[x,1] - xs[x,3]

                    # to calculate hydraulic radius perimeter is in the denominator so it has not to be zero
                    if xs[x,1] == 0 or xs[x,5] == 0:
                        assert False, 'at x= ' + str(Sub.xsid[x]) + ' t= ' + str(t) + ' total perimiter is zero'

                #for each cross section -------------------------------
                for x in range(0, Sub.XSno) :
                    #print(*,'(a3,i4,a4,i6,a]'] 't= ',t,' x= ', xsid[x],'------------------'
                    # forward difference
                    # surface slope is calculated based on the previous time step water depth
                    # except for the first cross section hx[1]is from current time step
                    # while hx[2]is from previous time step
                    if x < Sub.XSno-1:
                        #friction slope = (so - dhx/dx] - diffusive wave
                        sf = (hx[x] + Sub.bedlevel[x] - hx[x+1] - Sub.bedlevel[x+1])/dx
                    else:
                        #for LOWER BOUNDARY node
                        sf = (hx[x-1] + Sub.bedlevel[x-1] - hx[x] - Sub.bedlevel[x])/dx


         			#if sf <= 0 :
                        #print[800000,'(a]'] '----------'
                        #print(*,'(a4,i4,2x,a6,G10.4]'] 'XS= ',xsid[x], ', sf= ', sf
                        #print[800000,'(a7,i4,a5,i6,2x,a6,G10.4]'] 'min = ',t ,' XS= ',xsid[x], ', sf= ', sf


                    if River.D1['ModelMode'] == 2 :
                        #check for the dike overtopping
                        #to LEFT?
                        # overtoping dischargee is zero unless water level exceeds the dike level
                        q_outLx[x] = 0
                        # check first the water level in the 2d grid
                        # if water level in the 1D river is less than the water depth in
                        # nighbouring cell in the 2D grid don't calculate overtopping

                        if storewl[Sub.xsid[x]] - Sub.bedlevel[x]  <  hx[x] :

                            # if water depth is higher than the left dike height and the threshold
                            if  hx[x] > max(Sub.zl[x] - Sub.bedlevel[x], River.D1['MinDepth']):
                                #weir formular - free flow
                                q_outLx[x] = CWEIR*BWIDTH * (hx[x] - max(Sub.zl[x] - Sub.bedlevel[x] , River.D1['MinDepth']))**1.5

                        #to RIGHT?
                        q_outRx[x] = 0
                        # nxs is added to the ipf just because of the numbering system of the
                        # cross section
                        if  storewl[Sub.xsid[x] + River.XSno] - Sub.bedlevel[x] < hx[x] :

                            if hx[x] > max(Sub.zr[x] - Sub.bedlevel[x],River.D1['MinDepth']) :
                                q_outRx[x] = CWEIR * BWIDTH * (hx[x] - max( Sub.zr[x] - Sub.bedlevel[x],River.D1['MinDepth']) )**1.5



                        # correction factor
                        q_outx[x] = q_outLx[x] + q_outRx[x]
                        if q_outx[x] == 0:
                            corrL=0
                            corrR=0
                        else :
                            corrL = q_outLx[x]/q_outx[x]
                            corrR = q_outRx[x]/q_outx[x]



                    # compute mean area and hydraulic radius
                    # all cross section except the last one
                    if x < Sub.XSno-1:
                        if hx[x] <= Sub.Dbf[x]:
                            # upper discharge
                            xs[x,6] = 0
                            # diffusive wave coefficient for upper wave
                            Diff_coeff[x,0] = 0
                            # calculate only lower part
                            # Lower mean area and hydraulic radius
                            Area_LM = (xs[x,4] + xs[x+1,4])/2							#mean area
                            R_LM = ( xs[x,4]/xs[x,5] + xs[x+1,4]/xs[x+1,5] )/2		#mean radius
                        else :
                            # Upper mean area and hydraulic radius
                            Area_UM = (xs[x,2] + xs[x+1,2])/2						#mean area
                            #print(*,'(a7,f8.3]'] 'A_UM = ', Area_UM
                            if xs[x,3] != 0 and xs[x+1,3] == 0 :
                                R_UM = (xs[x,2]/xs[x,3])/2			#mean radius
                            elif xs[x,3] == 0 and xs[x+1,3] != 0:
                                R_UM = (xs[x+1,2]/xs[x+1,3])/2		#mean radius
                            elif xs[x,3] == 0 and xs[x+1,3] == 0:
                                R_UM = 0							#mean radius
                            else :
                                R_UM = (xs[x,2]/xs[x,3] + xs[x+1,2] / xs[x+1,3])/2		#mean radius

                            # upper discharge
                            xs[x,6] = (1.0/Sub.mn[x]) * Area_UM * (R_UM**power)
                            # diffusive wave coefficient for upper wave
                            Diff_coeff[x,0] = (1.0/Sub.mn[x])* (R_UM**power )

                            # Lower mean area and hydraulic radius
                            Area_LM = (xs[x,4]+xs[x+1,4])/2							#mean area
                            R_LM = (xs[x,4]/xs[x,5]+xs[x+1,4]/xs[x+1,5])/2		#mean radius
                    else : # the last cross section
                        if hx[x] <= Sub.Dbf[x]:
                            # upper discharge
                            xs[x,6] = 0
                            # diffusive wave coefficient for upper wave
                            Diff_coeff[x,0] = 0
                            # calculate only lower part
                            # Lower mean area and hydraulic radius
                            Area_LM = (xs[x,4]+xs[x-1,4])/2						#for LOWER BOUNDARY node
                            R_LM = (xs[x,4]/xs[x,5]+xs[x-1,4]/xs[x-1,5])/2		#mean radius
                        else : #for LOWER BOUNDARY node
                            # Upper mean area and hydraulic radius
                            Area_UM = (xs[x,2]+xs[x-1,2])/2

                            if xs[x,3] != 0 and xs[x-1,3] == 0:
                                R_UM = (xs[x,2]/xs[x,3])/2		#mean radius
                            elif xs[x,3] == 0 and xs[x-1,3] != 0:
                                R_UM = (xs[x-1,2]/xs[x-1,3])/2		#mean radius
                            elif xs[x,3] == 0 and xs[x-1,3] == 0:
                                R_UM = 0
                            else :
                                R_UM = (xs[x,2]/xs[x,3] + xs[x-1,2]/xs[x-1,3])/2	#mean radius

                            # upper discharge
                            xs[x,6] = (1.0/Sub.mn[x]) * Area_UM * (R_UM**power)
                            # diffusive wave coefficient for upper wave
                            Diff_coeff[x,0] = (1.0/Sub.mn[x])* (R_UM**power)

                            # Lower mean area and hydraulic radius
                            Area_LM = (xs[x,4]+xs[x-1,4])/2						#for LOWER BOUNDARY node
                            R_LM = (xs[x,4]/xs[x,5]+xs[x-1,4]/xs[x-1,5])/2		#mean radius


                    #calculation of fluxes
                    if abs(sf) > HLIN/dx:
                        # diffusive wave coefficient for upper wave
                        Diff_coeff[x,0] = Diff_coeff[x,0] / np.sqrt(abs(sf))
                        # diffusive wave coefficient for Lower wave
                        Diff_coeff[x,1] = (1.0/Sub.mn[x])* (R_LM**power) / np.sqrt(abs(sf))
                        # total diffusion coefficient
                        Diff_coeff[x,2] = Diff_coeff[x,0] + Diff_coeff[x,1]
                        # discharge
                        # upper discharge
                        xs[x,6] = xs[x,6] * (np.sqrt(abs(sf))*sf/abs(sf))
                        # Lower discharge
                        xs[x,7] = (1.0/Sub.mn[x]) * Area_LM * (R_LM**power) * (np.sqrt(abs(sf))*sf/abs(sf))
                    else :
                        # diffusive wave coefficient for upper wave
                        Diff_coeff[x,0] = Diff_coeff[x,0] / np.sqrt(HLIN/dx)

                        # diffusive wave coefficient for Lower wave
                        Diff_coeff[x,1] = (1.0/Sub.mn[x])* (R_LM**power) / np.sqrt(HLIN/dx)
                        # total diffusion coefficient
                        Diff_coeff[x,2] = Diff_coeff[x,0] + Diff_coeff[x,1]

                        # discharge
                        # upper discharge
                        xs[x,6] = xs[x,6] * (np.sqrt(dx/HLIN)*sf)
                        #xs[x,7] = xs[x,7] * (np.sqrt(HLINQ/dx]]
                        # Lower discharge
                        xs[x,7] = (1.0/Sub.mn[x]) * Area_LM * (R_LM**power) * (np.sqrt(dx/HLIN)*sf)
                        #xs[x,8] = (1.0/mn[x]] * Area_LM * (R_LM** (2.0/3.0]] * (np.sqrt(HLINQ/dx]]


                    # total Discharge
                    qx[x] = xs[x,6] + xs[x,7]

        # 			if qx[x] < 0:
                        #print(*,'(a,i4]'] 'Q is -ve at XS= ',xsid[x]
                        #print[800000,'(a7,i4,a17,i6,a4,f8.3]'] 'min = ',t ,' Q is -ve at XS= ',xsid[x],' Q= ',qx[x]
                        #print[800000,'(a,f8.3]'] 'Q is -ve at Q= ',qx[x]
                        #exit

                    if River.D1['ModelMode'] == 2:
                        # if the overtopping is less than 0.5 the flow
                        if qx[x] > 2*q_outx[x] :
                            # the whole overtopping will be considered
                            qx[x] = qx[x]- q_outx[x]
                        else :
                            # if the overtopping is greater than 0.5 the flow
                            # overtopping will be 0.5 the flow only
                            q_outx[x] = 0.5*qx[x]
                            q_outLx[x] = corrL*q_outx[x]
                            q_outRx[x] = corrR*q_outx[x]
                            qx[x] = qx[x]- q_outx[x]


                #for each cross section -------------------------------
                #print(*,'(a17,f8.3]'] 'max diff coeff = ', maxval[Diff_coeff(:,3]]

                # adjusting the time step
                dto = (dx**2)/(2*max(Diff_coeff[:,2]))

                if dto > dt:
                    dto = dt
                else :
                    if dto == 0:
                        print('adaptive time step is almost zero dto= '+ str(t))
                        #arr = Diff_coeff(:,3]
                        #call Find_FInArrayF[maxval[arr],arr,loc1[1]]
                        #loc =maxloc (Diff_coeff(:,3]]
                        #print(*,'(a37,i4]'] 'At XS= ', xsid[loc1[1]]
                        #print(*,'(a37,i4]'] 'Diff coeff= ', Diff_coeff[loc1[1],3]
                        #print(*,'(a37,i4]'] 'discharge= ', qx[loc1[1]]
                        assert False ,'dto= ' + str(dto)




                if dto > dtrest:
                    dto = dtrest

                #print[800000,'(a3,f8.2,a7,f4.2,a7,f8.2]'] 'dto= ', dto,' ratio=', dto/dt, ' max Q= ',qx[Sub.XSno]

                # update the water level at each cross section except the upstream boundary node
                for x in range(1, (Sub.XSno)-2) :
                    # check if the XS has laterals
                    # FindInArrayF(real[xsid[x]],sub_XSLaterals,loc)
                    if Sub.xsid[x] in Sub.LateralsTable:
                        loc = Sub.LateralsTable.index(Sub.xsid[x])
                        # have only the laterals
                        qlat = Lateral_q[t,loc]
                        #print(*,*] "Lat= ", qlat
                    else :
                        qlat = 0

                    # Forward time backward scpace
                    XSarea[x] = xs[x,0] - ((qx[x] - qx[x-1]) * (dto/dx)) + qlat * (dto/dx) #dto # difference between two consecutive XS areas

                    if XSarea[x] <= 0:
                        break
                        print(x)
                        print(XSarea[x])
                        assert False, "Error cross section area is less than zero"
                    else :
                        # calculate the bankful area
                        bankful_area = Sub.Dbf[x] * Sub.mw[x]

                        # if the new area is less than the area of the min[hl,hr]
                        if XSarea[x] <= Sub.AreaPerLow[x,0] :
                            # if the area is less than the bankful area
                            if XSarea[x] <= bankful_area:
                                hx[x] = XSarea[x]/Sub.mw[x]
                            else :
                                # Case 1
                                dummyarea = XSarea[x] - bankful_area
                                dummyh = SaintVenant.QuadraticEqn(Sub.cr[x]/Sub.hr[x] + Sub.cl[x]/Sub.hl[x], 2*Sub.mw[x],-2*dummyarea)
                                hx[x] = dummyh + Sub.Dbf[x]

                        # if the new area is less than the area of the max[hl,hr]
                        elif XSarea[x] <= Sub.AreaPerHigh[x,1]:
                            dummyarea = XSarea[x] - bankful_area
                            # check which dike is higher
                            if Sub.hl[x] < Sub.hr[x] :
                                # case 2
                                dummyh = SaintVenant.QuadraticEqn(Sub.cr[x]/Sub.hr[x]*0.5, Sub.mw[x]+Sub.cl[x],-dummyarea-Sub.hl[x]*Sub.cl[x]*0.5)
                            else :
                                # case 3
                                dummyh = SaintVenant.QuadraticEqn(Sub.cl[x]/Sub.hl[x]*0.5,Sub.mw[x]+Sub.cr[x],-dummyarea-Sub.hr[x]*Sub.cr[x]*0.5)

                            hx[x] = dummyh + Sub.Dbf[x]

                        else :
                            dummyarea = XSarea[x] - bankful_area
                            # case 4 (linear function]
                            dummyh = (dummyarea+Sub.cl[x]*Sub.hl[x]/2.0+Sub.cr[x]*Sub.hr[x]/2.0)/(Sub.cr[x]+Sub.cl[x]+Sub.mw[x])
                            hx[x] = dummyh + Sub.Dbf[x]



                        # to prevent -ve Sf in the next time step calculation
                        if hx[x]+Sub.bedlevel[x] > hx[x-1]+Sub.bedlevel[x-1]:
                            #print[800000,'(a]'] '-----'
                            #print[800000,'(a,i5,a11,i2]'] 'time = ',t , ' round no= ',adaptiveTS_counter
                            #print[800000,'(a9,i5,a4,f10.3]'] 'Q at x = ', xsid[x],' Q= ' ,qx[x]
                            #print[800000,'(a9,i5,a4,f10.3]'] 'Q at x = ', xsid[x-1],' Q= ' ,qx[x-1]
                            # g = (hx[x]+Sub.bedlevel[x])- (hx[x-1]+Sub.bedlevel[x-1])
                            #print[800000,'(a,i5,a3,f10.4,a2]'] 'H is higher than previous XS= ',xsid[x], "by ", g,' m'
                            #print(*,'(a,i5]'] 'H is higher than previous XS= ',xsid[x]
                            hx[x] = hx[x-1] + Sub.bedlevel[x-1] - Sub.bedlevel[x] -0.01


                        # this check to prevent the previous step from creating -ve values
                        # as after calculating the h[x]by solving the quadratic equation (QuadraticEqn]the step
                        # of hx[x] = hx[x-1] + bedlevel[x-1] - bedlevel[x] -0.01 can result in -ve hx
                        # so this check should be after not before
                        if hx[x] <= 0.01:
                            hx[x] = 0.01


                # water depth at the last point equals to wl at point before the last
                hx[Sub.XSno-1] = hx[Sub.XSno-2]- 0.001

                if min(hx)<0:
                    assert False ,'calculated water depth is less than 0 '

                # update the discharge array & dt
                dtrest = dtrest - dto
                q[:,t] = q[:,t] + qx[:]*(dto/dt)

                if River.D1['ModelMode'] == 2:
                    q_outL[:,t] = q_outL[:,t] + q_outLx* (dto/dt)
                    q_outR[:,t] = q_outR[:,t] + q_outRx* (dto/dt)
                    q_out[:,t] = q_outL[:,t] + q_outR[:,t]

            # adaptive time step ------------------------------------------------

            # store the calculated water level & water depth
            h[:,t] = hx
            wl = hx + Sub.bedlevel

        return q, h, wl


    def QuadraticEqn(a,b,c):
        delta = (b**2) - 4*a*c
        return (-b + np.sqrt(delta))/(2*a)

    def Kinematic(Model):
        beta = 3/5
        dx = Model.CellSize
        dt = 24*60*60
        dtx = dt/dx
        # for the river cells
        for i in range(Model.rows):
            for j in range(Model.cols):
                if not np.isnan(Model.FlowAccArr[i,j]) and Model.BankfullDepth[i,j] > 0:
                    p = Model.RiverWidth[i,j] + 2 * Model.BankfullDepth[i,j]
                    n = Model.RiverRoughness[i,j]
                    alpha1 = n * pow(p, 2/3)

                    # get the hydrograph of the upstream cells
                    UScells = Model.FDT[str(i) + "," + str(j)]
                    hyd = np.zeros(Model.TS, dtype=np.float32)
                    Laterals = np.zeros(Model.TS, dtype=np.float32)

                    # get the US cells and check which is river and which is lateral
                    rivercells = list()
                    lateralcells = list()
                    f = 0
                    for k in range(len(UScells)):
                        if Model.BankfullDepth[UScells[k][0], UScells[k][1]] == 0:
                            lateralcells.append(UScells[k])
                        else:
                            f = f + 1
                            rivercells.append(UScells[k])
                    # if the beginning of a river
                    if f == 0:
                        # get the sum of laterals as one hydrograph and route it
                        s = 0
                        for k in range(len(lateralcells)):
                            s = s + (Model.DEM[lateralcells[k][0], lateralcells[k][1]] - Model.DEM[i,j] ) / dx
                            hyd = hyd + Model.quz_routed[lateralcells[k][0], lateralcells[k][1],:]
                        # get average slope
                        s = s / len(lateralcells)
                        if s < HLIN/dx :
                            s = HLIN/dx
                        alpha = pow(alpha1 / pow(s,0.5) ,0.6)
                        Q = np.zeros(Model.TS, dtype=np.float32)
                        Q[0] = Model.quz[i,j,0]

                        for t in range(1,Model.TS):
                            val1 = alpha * beta * pow((Q[t-1] + hyd[t])/2,beta-1)
                            Q[t] = (dtx * hyd[t] +  val1 * Q[t-1] ) / (dtx + val1)

                        Model.quz_routed[i,j,:] = Q
                    else:
                        # if not the beginning of a river sum lateras and take them as one lateral time series
                        for k in range(len(lateralcells)):
                            Laterals = Laterals + Model.quz_routed[lateralcells[k][0], lateralcells[k][1],:]

                        for k in range(f):
                            # print(rivercells[k])
                            hyd = hyd + Model.quz_routed[rivercells[k][0], rivercells[k][1],:]
                            s = (Model.DEM[rivercells[k][0], rivercells[k][1]] - Model.DEM[i,j] ) / dx

                            if s < HLIN/dx :
                                s = HLIN/dx

                            alpha = pow(alpha1 / pow(s,0.5) ,0.6)
                            Q = np.zeros(Model.TS, dtype=np.float32)
                            Q[0] = Model.quz[i,j,0]

                            for t in range(1,Model.TS):
                                val1 = alpha * beta * pow((Q[t-1] + hyd[t])/2,beta-1)
                                Q[t] = (dtx * hyd[t] +  val1 * Q[t-1] + (Laterals[t] + Laterals[t-1])/2) / (dtx + val1)

                            Model.quz_routed[i,j,:] = Model.quz_routed[i,j,:] + Q