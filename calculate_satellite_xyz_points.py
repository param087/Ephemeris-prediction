import math
import datetime
import pandas as pd
df1 = pd.read_csv('pred_25_1_prn_2.csv')
df = pd.read_csv('org_25_1_prn_2.csv')

ti1 = []
ti2 = []
x1 = []
x2 = []
y1 = []
y2 = []
z1 = []
z2 = []
tkl1 = []
ekl1 = []
takl1 = []
toel1 = [] 
xpkl1 = []
ypkl1 = []
rkl1 = [] 
ikl1 = []
ukl1 = []
phikl1 = []
corr_il1 = []
corr_rl1 = []
corr_ul1 = []
vkl1 = []
vkl2 = []
tkl2 = []
ekl2 = []
takl2 = []
toel2 = [] 
xpkl2 = []
ypkl2 = []
rkl2 = [] 
ikl2 = []
ukl2 = []
phikl2 = []
omega1 = []
omega2 = []
corr_il2 = []
corr_rl2 = []
corr_ul2 = []
bPI =  3.1415926535898
bGM84 =  3.986005e14
bOMEGAE84 =  7.2921151467e-5
myprn = 2
earthrate = bOMEGAE84;
for i in range(len(df)):
    if df['prn'][i]==myprn:
        roota = df['sqrt_semi_major_axis'][i]
        t = df['time_of_ephemeris'][i]%86400.00;
        e = df['essentricity'][i]
        i0 = df['inclination'][i]/bPI
        smallomega = df['omega'][i]/bPI
        bigomega0 = df['OMEGA'][i]/bPI
        m0 = df['mean_anomaly'][i]/bPI

        toe = df['time_of_ephemeris'][i]%86400


        cus = df['correction_latitude_sine'][i]
        cuc = df['correction_latitude_cosine'][i]
        crs = df['correction_radius_sine'][i]
        cis = df['correction_inclination_sine'][i]
        cic = df['correction_inclination_cosine'][i]
        crc = df['correction_radius_cosine'][i]
        delta_n = df['mean_motion'][i]
        bigomegadot = df['OMEGA_dot'][i]/bPI
        idot = df['inclination_rate'][i]/bPI

        A = roota*roota;           #roota is the square root of A
        n0 = math.sqrt(bGM84/(A*A*A));  #bGM84 is what the ICD-200 calls Greek mu
        tk = t - toe;              #t is the time of the pos. & vel. request.
        n = n0 + delta_n;
        mk = m0 + n*tk;
        mkdot = n;
        ek = mk;
        for itera in range(0,2):
            ek = mk + e*math.sin(ek);  #Overkill for small e
        ekdot = mkdot/(1.0 - e*math.cos(ek));

        # True anomaly
        sinvk = (math.cos(ek)-e)/(1-e*math.cos(ek))
        cosvk = (math.sqrt(1-e*e)*math.sin(ek))/(1-e*math.cos(ek))
        vk = math.asin(sinvk)
        # phi


        #In the line, below, tak is the true anomaly (which is nu in the ICD-200).
        tak = math.atan2( math.sqrt(1.0-e*e)*math.sin(ek), math.cos(ek)-e);
        # takdot = math.sin(ek)*ekdot*(1.0+e*math.cos(tak))/(math.sin(tak)*(1.0-e*math.cos(ek)));

        phik = tak + smallomega; 
        # phik = vk + smallomega
        corr_u = cus*math.sin(2.0*phik) + cuc*math.cos(2.0*phik);
        corr_r = crs*math.sin(2.0*phik) + crc*math.cos(2.0*phik);
        corr_i = cis*math.sin(2.0*phik) + cic*math.cos(2.0*phik);
        
        uk = phik + corr_u;
        rk = A*(1.0-e*math.cos(ek)) + corr_r;
        ik = i0 + idot*tk + corr_i;

        
        # ukdot = takdot +2.0*(cus*math.cos(2.0*uk)-cuc*math.sin(2.0*uk))*takdot;
        # rkdot = A*e*math.sin(ek)*n/(1.0-e*math.cos(ek)) + 2.0*(crs*math.cos(2.0*uk)-crc*math.sin(2.0*uk))*takdot;
        # ikdot = idot + (cis*math.cos(2.0*uk)-cic*math.sin(2.0*uk))*2.0*takdot;

        xpk = rk*math.cos(uk);
        ypk = rk*math.sin(uk);

        # xpkdot = rkdot*math.cos(uk) - ypk*ukdot;
        # ypkdot = rkdot*math.sin(uk) + xpk*ukdot;

        omegak = bigomega0 + (bigomegadot-earthrate)*tk - earthrate*toe;

        # omegakdot = (bigomegadot-earthrate);

        xk = xpk*math.cos(omegak) - ypk*math.sin(omegak)*math.cos(ik);
        yk = xpk*math.sin(omegak) + ypk*math.cos(omegak)*math.cos(ik);
        zk = ypk*math.sin(ik);

        # xkdot = ( xpkdot-ypk*math.cos(ik)*omegakdot )*math.cos(omegak)- ( xpk*omegakdot+ypkdot*math.cos(ik)-ypk*math.sin(ik)*ikdot )*math.sin(omegak);
        # ykdot = ( xpkdot-ypk*math.cos(ik)*omegakdot )*math.sin(omegak)+ ( xpk*omegakdot+ypkdot*math.cos(ik)-ypk*math.sin(ik)*ikdot )*math.cos(omegak);
        # zkdot = ypkdot*math.sin(ik) + ypk*math.cos(ik)*ikdot;

        #print(xk, yk, zk)
        
        #Results follow.
        #print("BCpos: t, xk, yk, zk: %9.3Lf %21.11Lf %21.11Lf %21.11Lf\n", t, xk, yk, zk );
        #print(xk, yk, zk)
        """d = datetime.strptime(df1['epoch_time'][i], '%Y-%m-%d %H:%M:%S')
        datetime.strftime(d,'%y %m %d %H %M %S')"""
        if(i<100):
            print(" I : ",i)
            print(str(tk),str(ek),str(tak),str(toe),df['epoch_time'][i])
            print(xpk,ypk,uk,rk,ik,phik,corr_u,corr_i,corr_r)
            # print(str(float(df1['sqrt_semi_major_axis'][i]*df1['sqrt_semi_major_axis'][i])),df1['essentricity'][i],df1['inclination'][i]*180/math.pi,df1['OMEGA'][i]*180/math.pi,df1['omega'][i]*180/math.pi,df1['mean_anomaly'][i]*180/math.pi)
            print()

        tkl1.append(tk)
        ekl1.append(ek)
        takl1.append(tak)
        vkl1.append(vk)
        toel1.append(toe)
        xpkl1.append(xpk)
        ypkl1.append(ypk)
        rkl1.append(rk)
        ikl1.append(ik)
        ukl1.append(uk)
        phikl1.append(phik)
        corr_rl1.append(corr_r)
        corr_ul1.append(corr_u)
        corr_il1.append(corr_i)
        ti1.append(df['epoch_time'][i])
        x1.append(xk)

        omega1.append(omegak)
        y1.append(yk)
        z1.append(zk)

for i in range(len(df1)):    
    if df1['prn'][i]==myprn:
        roota = df1['sqrt_semi_major_axis'][i]
        t = df['time_of_ephemeris'][i]%86400.00;
        e = df1['essentricity'][i]
        i0 = df1['inclination'][i]/bPI
        smallomega = df1['omega'][i]/bPI
        bigomega0 = df1['OMEGA'][i]/bPI
        m0 = df1['mean_anomaly'][i]/bPI

        toe = df['time_of_ephemeris'][i]%86400


        cus = df1['correction_latitude_sine'][i]
        cuc = df1['correction_latitude_cosine'][i]
        crs = df1['correction_radius_sine'][i]
        cis = df1['correction_inclination_sine'][i]
        cic = df1['correction_inclination_cosine'][i]
        crc = df1['correction_radius_cosine'][i]
        delta_n = df1['mean_motion'][i]
        bigomegadot = df1['OMEGA_dot'][i]/bPI
        idot = df1['inclination_rate'][i]/bPI

        A = roota*roota;           #roota is the square root of A
        n0 = math.sqrt(bGM84/(A*A*A));  #bGM84 is what the ICD-200 calls Greek mu
        tk = t - toe;              #t is the time of the pos. & vel. request.
        n = n0 + delta_n;
        mk = m0 + n*tk;
        mkdot = n;
        ek = mk;
        for itera in range(0,2):
            ek = mk + e*math.sin(ek);  #Overkill for small e
        ekdot = mkdot/(1.0 - e*math.cos(ek));

        # True anomaly
        sinvk = (math.cos(ek)-e)/(1-e*math.cos(ek))
        cosvk = (math.sqrt(1-e*e)*math.sin(ek))/(1-e*math.cos(ek))
        vk = math.asin(sinvk)
        # phi


        #In the line, below, tak is the true anomaly (which is nu in the ICD-200).
        tak = math.atan2( math.sqrt(1.0-e*e)*math.sin(ek), math.cos(ek)-e);
        # takdot = math.sin(ek)*ekdot*(1.0+e*math.cos(tak))/(math.sin(tak)*(1.0-e*math.cos(ek)));

        phik = tak + smallomega;
        # phik = vk + smallomega
        corr_u = cus*math.sin(2.0*phik) + cuc*math.cos(2.0*phik);
        corr_r = crs*math.sin(2.0*phik) + crc*math.cos(2.0*phik);
        corr_i = cis*math.sin(2.0*phik) + cic*math.cos(2.0*phik);
        
        uk = phik + corr_u;
        rk = A*(1.0-e*math.cos(ek)) + corr_r;
        ik = i0 + idot*tk + corr_i;

        
        # ukdot = takdot +2.0*(cus*math.cos(2.0*uk)-cuc*math.sin(2.0*uk))*takdot;
        # rkdot = A*e*math.sin(ek)*n/(1.0-e*math.cos(ek)) + 2.0*(crs*math.cos(2.0*uk)-crc*math.sin(2.0*uk))*takdot;
        # ikdot = idot + (cis*math.cos(2.0*uk)-cic*math.sin(2.0*uk))*2.0*takdot;

        xpk = rk*math.cos(uk);
        ypk = rk*math.sin(uk);

        # xpkdot = rkdot*math.cos(uk) - ypk*ukdot;
        # ypkdot = rkdot*math.sin(uk) + xpk*ukdot;

        omegak = bigomega0 + (bigomegadot-earthrate)*tk - earthrate*toe;

        # omegakdot = (bigomegadot-earthrate);

        xk = xpk*math.cos(omegak) - ypk*math.sin(omegak)*math.cos(ik);
        yk = xpk*math.sin(omegak) + ypk*math.cos(omegak)*math.cos(ik);
        zk = ypk*math.sin(ik);

        # xkdot = ( xpkdot-ypk*math.cos(ik)*omegakdot )*math.cos(omegak)- ( xpk*omegakdot+ypkdot*math.cos(ik)-ypk*math.sin(ik)*ikdot )*math.sin(omegak);
        # ykdot = ( xpkdot-ypk*math.cos(ik)*omegakdot )*math.sin(omegak)+ ( xpk*omegakdot+ypkdot*math.cos(ik)-ypk*math.sin(ik)*ikdot )*math.cos(omegak);
        # zkdot = ypkdot*math.sin(ik) + ypk*math.cos(ik)*ikdot;

        #print(xk, yk, zk)
        
        #Results follow.
        #print("BCpos: t, xk, yk, zk: %9.3Lf %21.11Lf %21.11Lf %21.11Lf\n", t, xk, yk, zk );
        #print(xk, yk, zk)
        """d = datetime.strptime(df1['epoch_time'][i], '%Y-%m-%d %H:%M:%S')
        datetime.strftime(d,'%y %m %d %H %M %S')"""
        if(i<100):
            print(" I : ",i)
            print(str(tk),str(ek),str(tak),str(toe),df1['epoch_time'][i])
            print(xpk,ypk,uk,rk,ik,phik,corr_u,corr_i,corr_r)
            # print(str(float(df1['sqrt_semi_major_axis'][i]*df1['sqrt_semi_major_axis'][i])),df1['essentricity'][i],df1['inclination'][i]*180/math.pi,df1['OMEGA'][i]*180/math.pi,df1['omega'][i]*180/math.pi,df1['mean_anomaly'][i]*180/math.pi)
            print()

        tkl2.append(tk)
        ekl2.append(ek)
        takl2.append(tak)
        vkl2.append(vk)
        toel2.append(toe)
        xpkl2.append(xpk)
        ypkl2.append(ypk)
        rkl2.append(rk)
        ikl2.append(ik)
        ukl2.append(uk)
        phikl2.append(phik)
        corr_rl2.append(corr_r)
        corr_ul2.append(corr_u)
        corr_il2.append(corr_i)
        ti2.append(df1['epoch_time'][i])
        x2.append(xk)
        y2.append(yk)
        omega2.append(omegak)
        z2.append(zk)

df3 = pd.DataFrame()
df3['epoch_time1'] = ti1
df3['epoch_time2'] = ti2
df3['x_act'] = x1
df3['x_pred'] = x2
df3['y_act'] = y1
df3['y_pred'] = y2
df3['z_act'] = z1
df3['z_pred'] = z2
dx = []
dy = []
dz = []
for i in range(len(x1)):
    dx.append(math.sqrt((x1[i]-x2[i])**2))
    dy.append(math.sqrt((y1[i]-y2[i])**2))
    dz.append(math.sqrt((z1[i]-z2[i])**2))

re=[]
for i in range(len(x1)):
    re.append(math.sqrt(dx[i]**2+dy[i]**2+dz[i]**2))
df3['dx'] = dx
df3['dy'] = dy
df3['dz'] = dz
df3['radial_error'] = re
df3.to_csv('Output.csv')

# calculate the difference between the points
for i in range(4):
    print()
    print("x1 = "+str(x1[i]),"\t x2 = "+str(x2[i]),"\t diff = "+str(x1[i]-x2[i])+" \t ",ti1[i],ti2[i])
    print("y1 = "+str(y1[i]),"\t y2 = "+str(y2[i]),"\t diff = "+str(y1[i]-y2[i])+" \t ",ti1[i],ti2[i])
    print("z1 = "+str(z1[i]),"\t z2 = "+str(z2[i]),"\t diff = "+str(z1[i]-z2[i])+" \t ",ti1[i],ti2[i])
    print("final diff = ", ((x1[i]-x2[i])**2+(y1[i]-y2[i])**2+(z1[i]-z2[i])**2)**0.5)
# for i in range(len(x1)):
#     print("*********************************************************")
#     print(ti1[i],ti2[i])
#     print('tk ',tkl1[i],tkl2[i])
#     print('ek ',ekl1[i],ekl2[i])
#     print('tak ',takl1[i],takl2[i])
#     print('vk ',vkl1[i],vkl2[i])
#     print('toe ',toel1[i],toel2[i])
#     print('xpk ',xpkl1[i],xpkl2[i])
#     print('ypk ',ypkl1[i],ypkl2[i])
#     print('rk ',rkl1[i],rkl2[i])
#     print('ik ',ikl1[i],ikl2[i])
#     print('uk ',ukl1[i],ukl2[i])
#     print('omega ', omega1[i], omega2[i])
#     print('phik ',phikl1[i],phikl2[i])
#     print('cu ',corr_ul1[i],corr_ul2[i])
#     print('cr ',corr_rl1[i],corr_rl2[i])
#     print('ci ',corr_il1[i],corr_il2[i])
#     print("*********************************************************")
