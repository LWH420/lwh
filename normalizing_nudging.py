# 处理2度版本的逐步长/逐小时输出的数据，选取一定点，进行全场的标准化
# 修改nn_train_nudging.nml、输入输出文件   
import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import dask.array as da
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter , LatitudeFormatter
import cartopy.feature as cfeat
import f90nml as fnl
import sys
from multiprocessing import Manager

exp_name =sys.argv[1]
#exp_name ="nudging_0.01_new"
inpath=r"/data2/share/llj/lwh/FGOALS_output/"+exp_name+"/run/"


year="2010"
mon ="01"
day_select =sys.argv[2]
date=year+"-"+mon+"_days_"+day_select
print(exp_name)

NN_NML_path="./nn_train_nudging.nml"
nml=fnl.read(NN_NML_path)["PRE_PROC_NML"]
select_num=nml["select_num"]
# nudging_alpha=nml['nudging_alpha']
# nudging_interval=nml['nudging_interval']
SECONDS_PER_DAY = 86400.0
# select_num=512  # global select numbers
rand_seed=10000
#%%
tropical_ratio=nml["tropical_ratio"]
subtrop_ratio=nml["subtrop_ratio"]
polar_ratio=nml["polar_ratio"]

area_split="trop_{}.subtrop_{}.polar_{}".format(tropical_ratio,subtrop_ratio,polar_ratio)
file_name_used="Resampling_selectnum_"+str(select_num)+"_randomseed_"+str(rand_seed)+"_"+str(area_split)+"_nudging_plot"

def plot_grid(lon_grid,lat_grid,file_name):
    font1={'weight':'normal','size':20}
    fig = plt.figure( figsize =(10 , 6) )
    axs = fig.add_subplot(111 , projection = ccrs.PlateCarree ( central_longitude=180) )
    prj = ccrs.PlateCarree( central_longitude =0)
    axs.scatter(lon_grid[:] ,lat_grid[:] , 40, marker='.',color="r", transform = prj )
    axs.set_extent ([0 , 357.5 , -90 , 90] , prj )
    axs.set_xticks ( range (0, 360 , 30) , crs = prj )
    axs.set_yticks ( range (-90 , 90 , 30) , crs = prj )
    xtick=LongitudeFormatter( zero_direction_label = True )
    ytick=LatitudeFormatter()
    axs.xaxis.set_major_formatter(xtick)
    axs.yaxis.set_major_formatter(ytick)
    axs. add_feature( cfeat.COASTLINE )
    axs.set_xlabel('Longitude',font1)
    axs.set_ylabel('Latitude',font1)
    axs.grid()
    axs.tick_params(labelsize=15)
    # axs.set_title('Selected Locations',font1)
    plt.savefig(file_name+".pdf")
    # plt.show()

def get_subsample_index(lon,lat):
    x_mesh,y_mesh=np.meshgrid(lon,lat)


    print(y_mesh)
    print(y_mesh.shape)  # 600*1440
    # exit()
    np.random.seed(rand_seed)

    x_mesh_1d=x_mesh.reshape(-1)
    y_mesh_1d=y_mesh.reshape(-1)
    ##2025.12.9 modifeied by lwh
    ind1=np.where((y_mesh_1d > -90) & (y_mesh_1d <= -75))[0]
    ind2=np.where((y_mesh_1d > -75) & (y_mesh_1d <= -25))[0]
    ind3=np.where((y_mesh_1d > -25) & (y_mesh_1d <=  0 ))[0]
    ind4=np.where((y_mesh_1d >  0 ) & (y_mesh_1d <=  25))[0]
    ind5=np.where((y_mesh_1d >  25) & (y_mesh_1d <=  75))[0]
    ind6=np.where((y_mesh_1d >  75) & (y_mesh_1d <=  90))[0]
    print(ind1)
    index_select=np.array([-999.],int)
    print(index_select)
    index_select=np.append(index_select,np.random.choice(ind1,size=int(select_num*polar_ratio/2),replace=False))
    index_select=np.append(index_select,np.random.choice(ind2,size=int(select_num*subtrop_ratio/2),replace=False))
    index_select=np.append(index_select,np.random.choice(ind3,size=int(select_num*tropical_ratio/2),replace=False))
    index_select=np.append(index_select,np.random.choice(ind4,size=int(select_num*tropical_ratio/2),replace=False))
    index_select=np.append(index_select,np.random.choice(ind5,size=int(select_num*subtrop_ratio/2),replace=False))
    index_select=np.append(index_select,np.random.choice(ind6,size=int(select_num*polar_ratio/2),replace=False))
    index_select=index_select[1:]
    print("index_select shape is ",index_select.shape[0])
    print("index_select",index_select)
    # print(x_mesh_1d[26919:26921])    #41013  89503  26920  59394
    # print(y_mesh_1d[26919:26921])
    x_mesh_1d=x_mesh_1d[index_select]
    y_mesh_1d=y_mesh_1d[index_select]
    print(x_mesh_1d[:3])
    print(y_mesh_1d[:3])

    assert index_select.shape[0] == select_num, "index_select != select_num"
    plot_grid(x_mesh_1d,y_mesh_1d,file_name_used)

    return index_select,x_mesh_1d,y_mesh_1d

# samples_cut=5
def normalize_variable(variable_data):
    """
    对每个变量的数据进行逐层归一化。
    """
    # 计算最小值和最大值
    min_val = variable_data.min(dim=('time','grid'))
    max_val = variable_data.max(dim=('time','grid'))
    
    # 归一化处理 (每个变量的垂直层归一化)
    normalized_data = (variable_data - min_val) / (max_val - min_val)
    print(f"min is {min_val.values}, max is {max_val.values}")
    # 返回归一化后的数据和对应的最大最小值
    return normalized_data, min_val, max_val

def normalize_whole(variable_data):
    """
    对每个变量的数据进行归一化。
    """
    # 计算最小值和最大值
    min_val = variable_data.min()
    max_val = variable_data.max()
    
    # 归一化处理 (每个变量的垂直层归一化)
    normalized_data = (variable_data - min_val) / (max_val - min_val)
    print(f"min is {min_val.values}, max is {max_val.values}")
    # 返回归一化后的数据和对应的最大最小值
    return normalized_data, min_val, max_val


def process_variable(variable_name, data):
    """
    处理每个变量的归一化操作，并将结果保存到结果字典中。
    """
    normalized_data, min_val, max_val = normalize_variable(data) ##改成对每一层进行归一化
    #normalized_data, min_val, max_val = normalize_whole(data)
    # tendency = compute_nudging_tendency(data)
    print(f"Processing {variable_name} with data: {data}") 
    return variable_name,normalized_data, min_val, max_val
    
def parallel_normalization(input_file, output_file, var_3d_list=None,var_2d_list=None):
    """
    对 NetCDF 文件中的每个变量进行并行化处理，进行归一化。
    """
    if var_3d_list is None:
        var_3d_list = []
    if var_2d_list is None:
        var_2d_list = []
    # 读取 NetCDF 文件
    all_vars = list(var_3d_list) + list(var_2d_list)
    if len(all_vars) == 0:
        raise ValueError("没有传入任何需要归一化的变量")
    
    ds = xr.open_dataset(input_file,engine="netcdf4",decode_times=False)
    print(f"Dataset: {ds}")
    num_lon=ds.sizes['lon']
    num_lat=ds.sizes['lat']

    
    # num_ilev=ds.dims['ilev']
    # print(ds.dims['time'])
    num_time=ds.sizes['time']
        # num_time=time_split_end-time_split_start

    # print(ds)
    lon=ds.coords['lon']
    lat=ds.coords['lat']
    index_select,x_mesh_1d,y_mesh_1d=get_subsample_index(lon,lat)
        # da = ds[var] 
        # da_s = da.stack(grid=("lat", "lon")).isel(grid=index_select)
        # print("selected variable shape is ",da_s.shape)
        # 创建进程池
    print("start normalizing")
    tasks = []

    for var in all_vars:
        da = ds[var].stack(grid=("lat", "lon")).isel(grid=index_select)
        da = da.reset_index("grid", drop=True)
        da = da.assign_coords(grid=np.arange(da.sizes["grid"]))
        # 关键：自动判断是 3D 还是 2D
        if "lev" in da.dims:
            da = da.transpose("time", "lev", "grid")
        else:
            da = da.transpose("time", "grid")

        # multiprocessing 更稳一点：先 load 到内存
        tasks.append((var, da.load()))

    with multiprocessing.Pool(processes=min(len(all_vars), multiprocessing.cpu_count()),
                              maxtasksperchild=5) as pool:
        results = pool.starmap(process_variable, tasks)
    print("end normalizing")
    print("----------------------------------------------------")
    print(results)

      # 将结果合并到字典中
    ds_out = xr.Dataset()

    # 坐标
    ds_out = ds_out.assign_coords(time=ds["time"])
    ds_out = ds_out.assign_coords(grid=np.arange(len(index_select)))

    if len(var_3d_list) > 0 and "lev" in ds.coords:
        ds_out = ds_out.assign_coords(lev=ds["lev"])

    # 保存被抽样后的格点位置
    ds_out["lon_grid"] = xr.DataArray(
        x_mesh_1d.astype("float32"), dims=("grid",), attrs={"long_name": "selected longitude"}
    )
    ds_out["lat_grid"] = xr.DataArray(
        y_mesh_1d.astype("float32"), dims=("grid",), attrs={"long_name": "selected latitude"}
    )

    # 写入变量和 min/max
    for var_name, da_norm, vmin, vmax in results:
        ds_out[var_name] = da_norm
        ds_out[f"{var_name}_min"] = vmin
        ds_out[f"{var_name}_max"] = vmax

    ds_out.to_netcdf(output_file)

    print(f"Processed and saved normalized data to {output_file}")

if __name__ == "__main__":
    import glob
    import multiprocessing

    ntimes_input=2
    n_days=1
    infile=exp_name+r".gamil.h2."+year+"-"+mon+"-"+day_select+"-*"
    in_paths=inpath+infile
    # outpath=r"/lustre/gmmcfs/hjj/training_data/"
    cdo_outdir=r"/data2/share/llj/lwh/ML_code/lwh_NN_model/nudging_cdo_output/"
    outfile=exp_name+"."+date+".nc"
    cdo_out_paths=cdo_outdir+outfile
    print(cdo_out_paths)
    os.system(f"mkdir -p {cdo_outdir}")
    outpath=r"/data2/share/llj/lwh/ML_code/lwh_NN_model/nudging_normalized_output/"
    os.system(f"mkdir -p {outpath}")
    #%%
    # os.system("rm -r {out_paths}")
    # if no file
    #-------------------------------------------------------------------------------
    if not os.path.exists(cdo_out_paths) :
        print("cdo preprocessing.",flush=True)
        print(in_paths)
        print(cdo_out_paths)
        os.system(f'cdo -P 32 --no_history mergetime {in_paths} {cdo_out_paths}')
    # input_var_3d_lev    = ["ZM_DPP","ZM_PAP","ZM_PLAQ","ZM_PLAT","ZM_QH",\
                        #    "ZM_U","ZM_V","ZM_ZM"]
    #------------------------------------------------------------------------
        #%%
    input_var_3d_lev    = ["U","V","T","Q"] ##19
    if exp_name == "no_nudging_2deg":
        input_var_2d   = ["PHIS","LANDFRAC"]#None#["dsubcld"]
    else:
        input_var_2d   =None
    output_var_3d_lev   =   ["U","V","T","Q"]
    output_var_2d       =  None
    print(input_var_3d_lev)
    print(input_var_2d )
    print("input data preprocessing.",flush=True)

    #%%
    out_paths=outpath+exp_name+"_"+date+"_ntimesinput_"+str(ntimes_input)+"_inputdata_"+file_name_used+"lev_normalized.nc"
    print(out_paths)
    parallel_normalization(cdo_out_paths, out_paths, input_var_3d_lev,input_var_2d)
    out_paths=outpath+exp_name+"_"+date+"_outputdata_"+file_name_used+"lev_normalized.nc"
    print(out_paths)
    parallel_normalization(cdo_out_paths, out_paths, output_var_3d_lev,input_var_2d)
    #%%
    print("input data finish")