from Hapi.hm.river import River

path = "F:/01Algorithms/Hydrology/HAPI/examples/"
#%% create the River object
start = "2010-1-1 00:00:00"
end = "2010-1-1 05:00:00"
# rrm_start="1950-1-1 00:00:00"
# dx in meter
dx = 20
# dt in sec
dto = 50  # sec
Test = River(
    "Test", version=4, start=start, end=end, dto=dto, dx=dx, fmt="%Y-%m-%d %H:%M:%S"
)
Test.oneminresultpath = path + "/data/hydrodynamic model/"

Test.Time = 5 * 60 * 60  # (hrs to seconds)
# Read Input Data
Test.readXS(path + "/data/hydrodynamic model/xs_hz.csv")
#%% Initial and Boundary condition
Test.icq = 0
Test.ich = 12
Test.readBoundaryConditions(
    path=path + "/data/hydrodynamic model/BCH-2.txt",
    fmt="%Y-%m-%d %H:%M:%S",
    ds=True,
    dsbcpath=path + "/data/hydrodynamic model/BCQ-constant.txt",
)
#%% Run the model
start = "2010-1-1 00:00:00"
end = "2010-1-1 05:00:00"
Test.preissmann(
    start, end, fmt="%Y-%m-%d %H:%M:%S", maxiteration=5, beta=1, epsi=0.5, theta=0.55
)
print("Stability Factor = " + str(Test.stabilityfactor.min()))
#%% Visualization
start = "2010-01-01 00:00:00"
end = "2010-1-1 05:00:00"
# ffmpeg_path = "F:/Users/mofarrag/.matplotlib/ffmpeg-4.4-full_build/bin/ffmpeg.exe"
anim = Test.animatefloodwave(start=start, end=end, interval=2, textlocation=-1)
