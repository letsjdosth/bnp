from statistics import mean

def semi_variogram(data, long, lat):
    dist_u = []
    vario_r = []
    for i, (dat_i, long_i, lat_i) in enumerate(zip(data, long, lat)):
        for j, (dat_j, long_j, lat_j) in enumerate(zip(data, long, lat)):
            if i>j:
                pass
            else:
                dist_ij = ((long_i-long_j)**2 + (lat_i-lat_j)**2)**0.5
                vario_ij = 0.5 * (dat_i-dat_j)**2
                dist_u.append(dist_ij)
                vario_r.append(vario_ij)
    return dist_u, vario_r

def directional_semi_variogram(data, long, lat, direction):
    dist_u = []
    vario_r = []
    for i, (dat_i, long_i, lat_i) in enumerate(zip(data, long, lat)):
        for j, (dat_j, long_j, lat_j) in enumerate(zip(data, long, lat)):
            if i>j:
                pass
            else:
                if direction == 0 and long_i<long_j: #right
                    dist_ij = ((long_i-long_j)**2 + (lat_i-lat_j)**2)**0.5
                    vario_ij = 0.5 * (dat_i-dat_j)**2
                    dist_u.append(dist_ij)
                    vario_r.append(vario_ij)
                elif direction == 1 and long_i>long_j: #left
                    dist_ij = ((long_i-long_j)**2 + (lat_i-lat_j)**2)**0.5
                    vario_ij = 0.5 * (dat_i-dat_j)**2
                    dist_u.append(dist_ij)
                    vario_r.append(vario_ij)
                elif direction == 2 and lat_i<lat_j: #up
                    dist_ij = ((long_i-long_j)**2 + (lat_i-lat_j)**2)**0.5
                    vario_ij = 0.5 * (dat_i-dat_j)**2
                    dist_u.append(dist_ij)
                    vario_r.append(vario_ij)
                elif direction == 3 and lat_i>lat_j: #down
                    dist_ij = ((long_i-long_j)**2 + (lat_i-lat_j)**2)**0.5
                    vario_ij = 0.5 * (dat_i-dat_j)**2
                    dist_u.append(dist_ij)
                    vario_r.append(vario_ij)
                    
    return dist_u, vario_r

def binning(dist, vario, num_bins):
    max_u = max(dist)
    bin_tres_u = [(i+1)*max_u/num_bins for i in range(num_bins)]
    bin_middle_u = [(i+0.5)*max_u/num_bins for i in range(num_bins)]
    bin_vario_list = [[] for _ in range(num_bins)]
    # print(bin_tres_u)

    for u, r in zip(dist,vario):
        add_flag = False
        for i, tres in enumerate(bin_tres_u):
            if u<tres:
                bin_vario_list[i].append(r)
                add_flag = True
                break
        if not add_flag:
            bin_vario_list[-1].append(r)
    
    # print(bin_vario_list[-1])
    bin_vario_r = [mean(x) for x in bin_vario_list]
    # print(bin_vario_r)
    return bin_middle_u, bin_vario_r
