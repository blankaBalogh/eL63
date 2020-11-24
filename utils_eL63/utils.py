import numpy as np
import matplotlib.pyplot as plt


# -- Plotting Lorenz attractors in 3D with matplotlib
def plot_3d_timeseries(z_Bz, i0=0, cbar=False, cmap="", save=False, sdir="", 
        title="Lorenz", quiver=False, sformat="eps", scaled=True) :
    """
    Plots 3D timeseries.
    
    INPUT :
    z_Bz        Timeseries to plot.
    i0          Index of start time.            (Default : 0)
    cbar        Display colorbar ?              (Default : False)
    cmap        Colormap name.                  (Default : None, "black")
    save        Save resulting plot ?           (Default : False)
    sdir        Save directory.                 (Default : local, "./")
    title       Saved file name.                (Default : "Lorenz")
    sformat     Output file format.             (Default : eps)
    quiver      If quiver instead of axes.      (Default : False)
    scaled      Ax bounds set to eL63 default   (Default : True)
                if scaled=True.

    OUTPUT :
    (saved figure)

    """
    import matplotlib as mpl

    x,y,z = z_Bz[i0:,0], z_Bz[i0:,1], z_Bz[i0:,2]
    Nt_snapshots = x.shape[0]
    dt = 0.01
    time_lim = int(Nt_snapshots*dt)

    # Setting up figure
    fig = plt.figure(figsize=(10,10))
    #plt.subplots_adjust(top=.95)
    ax = plt.axes(projection='3d')

    # Setting up background features
    ax.set_facecolor('white')
    fig.set_facecolor('white')
    ax.grid(False) 

    ax.w_xaxis.pane.fill = False
    ax.w_yaxis.pane.fill = False
    ax.w_zaxis.pane.fill = False
    
    ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    #palette = sns.color_palette("rocket_r", as_cmap=True)

    # ax limits & names
    ax.set_xlim3d(x.min(), x.max())
    ax.set_xlabel('$x_1$')
    ax.set_ylim3d(y.min(), y.max())
    ax.set_ylabel('$x_2$')
    ax.set_zlim3d(z.min(), z.max())
    ax.set_zlabel('$x_3$')
    
    if scaled :
        ax.set_xlim3d(-20, 20)
        ax.set_ylim3d(-25,30)
        ax.set_zlim3d(0,50)
    
    if quiver : 
        plt.axis('off')
    
        xq = np.array([-20,-20,-20])
        yq = np.array([-20,-20,-20])
        zq = np.array([0,0,0])
        u, v, w = np.array([[7,0,0],[0,7,0],[0,0,7]])
        ax.quiver(xq,yq,zq,u,v,w,arrow_length_ratio=0.3, color="black")
        ax.text(-15, -22, -2.5, "$Z_1$")
        ax.text(-18, -16.5, 1, "$Z_2$")
        ax.text(-22, -22, 5, "$Z_3$")

    # Plotting data
    scatter = False
    if scatter :
        ax.scatter(x, y, z, s=0.5)
        if save :
            plt.savefig(sdir+'a_Lord3_3_Bz_scatter.png')
        
    else : 
        if cmap=="" :
            ax.plot3D(x[i0:], y[i0:], z[i0:], color="black", linewidth=1)
            
        else :
            points = np.array([x[i0:], y[i0:], z[i0:]]).T.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            cmap=plt.get_cmap(cmap)
            colors=[cmap(float(ii)/(Nt_snapshots-1)) for ii in range(Nt_snapshots-1)]
            
            ax.text(x[i0], y[i0], z[i0], "+", color='black', size=25, ha='center', va='center')
            for ii in range(Nt_snapshots-1):
                segii=segments[ii]
                lii,=ax.plot(segii[:,0], segii[:,1], segii[:,2], color=colors[ii],
                        linewidth=1)
                lii.set_solid_capstyle('round')
        
            if cbar :
                norm = mpl.colors.Normalize(vmin=0,vmax=time_lim)
                sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                cbar_ax = fig.add_axes([0.95, 0.35, 0.01, 0.5])
                cbar = plt.colorbar(sm, cax=cbar_ax)
                cbar.set_label('time (MTU)', rotation=90, fontsize=15, labelpad=30)
            
        if save :
            plt.savefig(sdir+title+"."+sformat, bbox_inches="tight", format=sformat)


# -- is_stable indicates which trajectories are (un)stable
def is_stable(trajectories, mini, maxi, print_nans=False) :
    """
    Print whether "trajectories" is diverging or not regarding (mini, maxi) values.

    INPUT :
    trajectories        Orbit coordinates.
    mini, maxi          Minimal & maximal divergence criteria.
    print_nans          Printing how many axes contains NaNs ?  (Default : False)

    """
    if print_nans :
        print("NaNs : ")
        print(np.isnan(np.max(trajectories, axis=(0,1))).astype(int).sum())
    
    if ((np.nanmax(trajectories, axis=(0,1)) > maxi).astype(np.int)).sum()==0 :
        if ((np.nanmax(trajectories, axis=(0,1))< mini).astype(np.int)).sum()==0 :
            return("Stable")
        else :
            return("Unstable")
    else :
        return("Stable")



def divergence(x_ML, mini, maxi) :
    minlist = list(np.unique(np.where((np.nanmax(x_ML, axis=1))>maxi)[0]))
    maxlist = list(np.unique(np.where((np.nanmin(x_ML, axis=1))<mini)[0]))

    list_divergence = list(dict.fromkeys(minlist + maxlist))
    
    return(list_divergence)

