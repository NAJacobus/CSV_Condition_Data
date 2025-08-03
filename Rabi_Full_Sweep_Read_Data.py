import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider
import matplotlib as mpl
import numpy as np
import math
from truncdec import truncdec

mpl.rcParams["text.usetex"] = True

file_name = "C:/Users/akash/PycharmProjects/HolsteinModel/P3_PPT_Rabi/Rabi_Full_Sweep_Data/2024-07-05_vib=10_Rabi_dense_3.npz"
data = np.load(file_name, allow_pickle=True)

#Plot Settings
#Choose between "om", "bs", "bb", "sc", "bc", "del", "lam", "t" to appear on the axes; the rest will appear as sliders
x_ax = "del"
y_ax = "lam"

model = data["model"][0]
spin_down_pops = data["spin_down_pops"]
purity_data = data["purity_data"]
numerical_negativity_data = data["numerical_negativity_data"]
stored_ee_t_rho = data["stored_ee_t_rho"]
stored_site_t_rho = data["stored_site_t_rho"]
stored_site_t_pt = data["stored_site_t_pt"]
omega_list = data["omega_list"]
bs_list = data["bs_list"]
bb_list = data["bb_list"]
sc_list = data["sc_list"]
bc_list = data["bc_list"]
deltas = data["deltas"]
lam_list = data["lam_list"]
t_list = data["t_list"]

sweep_indices = {}
sweep_indices["om"] = 0
sweep_indices["bs"] = 1
sweep_indices["bb"] = 2
sweep_indices["sc"] = 3
sweep_indices["bc"] = 4
sweep_indices["del"] = 5
sweep_indices["lam"] = 6
sweep_indices["t"] = 7

indices_to_vals = {}
indices_to_vals[0] = omega_list
indices_to_vals[1] = bs_list
indices_to_vals[2] = bb_list
indices_to_vals[3] = sc_list
indices_to_vals[4] = bc_list
indices_to_vals[5] = deltas
indices_to_vals[6] = lam_list
indices_to_vals[7] = t_list

indices_to_names= {}
indices_to_names[0] = r"$\omega$"
indices_to_names[1] = r"$\beta_s$"
indices_to_names[2] = r"$\beta_b$"
indices_to_names[3] = r"$\gamma_s$"
indices_to_names[4] = r"$\gamma_b$"
indices_to_names[5] = r"$\Delta$"
indices_to_names[6] = r"$\lambda$"
indices_to_names[7] = r"$t$"

y_ind = sweep_indices[y_ax]
x_ind = sweep_indices[x_ax]
remainder = list(range(8))
remainder.remove(x_ind)
remainder.remove(y_ind)
perm = tuple([y_ind, x_ind] + remainder)

numerical_negativity_original = np.copy(numerical_negativity_data)
for index, i in np.ndenumerate(numerical_negativity_data):
    if i <= 0: #Note that we already performed a cutoff in negativity_all_subdivisions when we only added to the negativity when it was greater than the threshold
        numerical_negativity_data[*index] = np.nan

spin_down_pops = np.transpose(spin_down_pops, perm)
purity_data = np.transpose(purity_data, perm)
numerical_negativity_data = np.transpose(numerical_negativity_data, perm)
numerical_negativity_original = np.transpose(numerical_negativity_original, perm)
stored_ee_t_rho = np.transpose(stored_ee_t_rho, perm)
stored_site_t_rho = np.transpose(stored_site_t_rho, perm)
stored_site_t_pt = np.transpose(stored_site_t_pt, perm)

default_backend = mpl.get_backend()


def get_index_from_val(val, val_list):
    index = 0
    while index + 1 < len(val_list) and val_list[index] < val:
        index += 1
    return index

def on_pick(event, slider_vals, subplot_axes):
    if event.inaxes in subplot_axes:
        print(indices_to_names[x_ind] + " = " + str(truncdec(event.xdata, 2)) + indices_to_names[y_ind] + " = " + str(truncdec(event.ydata, 2)))

        indexX = get_index_from_val(event.xdata, indices_to_vals[x_ind])
        indexY = get_index_from_val(event.ydata, indices_to_vals[y_ind])
        ind_0 = list(indices_to_vals[remainder[0]]).index(slider_vals[0])
        ind_1 = list(indices_to_vals[remainder[1]]).index(slider_vals[1])
        ind_2 = list(indices_to_vals[remainder[2]]).index(slider_vals[2])
        ind_3 = list(indices_to_vals[remainder[3]]).index(slider_vals[3])
        ind_4 = list(indices_to_vals[remainder[4]]).index(slider_vals[4])
        ind_5 = list(indices_to_vals[remainder[5]]).index(slider_vals[5])

        index_tuple = (indexY, indexX, ind_0, ind_1, ind_2, ind_3, ind_4, ind_5)
        # mpl.use(default_backend)
        fig1 = plt.figure()
        gs1 = gridspec.GridSpec(40, 40)
        axb = fig1.add_subplot(gs1[:])
        axb.axis("off")
        axb.set_title("Real Part " + model + " " + indices_to_names[x_ind] + " = " + str(truncdec(event.xdata, 2)) + ", " + indices_to_names[y_ind] + " = " + str(truncdec(event.ydata, 2)) + ", " + indices_to_names[remainder[0]] + " = " + str(truncdec(slider_vals[0],2)) + ", " + indices_to_names[remainder[1]] +  " = " + str(truncdec(slider_vals[1],2)) + ", " + indices_to_names[remainder[2]] + " = " + str(truncdec(slider_vals[2],2)) + ", " + str(indices_to_names[remainder[3]]) + " = " + truncdec(slider_vals[3],2) + ",\n " + indices_to_names[remainder[4]] + " = " + truncdec(slider_vals[4],2) + ", " + indices_to_names[remainder[5]] + " = " + str(truncdec(slider_vals[5],2)) + ", S.D.P. = " + str(truncdec(spin_down_pops[*index_tuple], 2)) + ", Purity = " + str(truncdec(purity_data[*index_tuple], 2)) + ", neg = " + str(truncdec(numerical_negativity_original[*index_tuple], 2)))
        gs1.update(wspace=0.5)
        ax4 = plt.subplot(gs1[5:20, 5:20])
        ax5 = plt.subplot(gs1[5:20, 25:40])
        ax6 = plt.subplot(gs1[25:40, 13:28])
        fig_4 = ax4.imshow(np.real(stored_ee_t_rho[*index_tuple]), interpolation='nearest')
        ax4.title.set_text("Real Part; Energy Eigenbasis")
        fig_5 = ax5.imshow(np.real(stored_site_t_rho[*index_tuple]), interpolation='nearest')
        ax5.title.set_text("Real Part; Product Basis")
        fig_6 = ax6.imshow(np.real(stored_site_t_pt[*index_tuple]), interpolation='nearest')
        ax6.title.set_text("Real Part; Partial Transpose")
        plt.colorbar(fig_4, ax=ax4)
        plt.colorbar(fig_5, ax=ax5)
        plt.colorbar(fig_6, ax=ax6)
        plt.show()

        fig2 = plt.figure()
        gs2 = gridspec.GridSpec(40, 40)
        axc = fig2.add_subplot(gs2[:])
        axc.axis("off")
        axc.set_title(
            "Imag Part " + model + " " + indices_to_names[x_ind] + " = " + str(truncdec(event.xdata, 2)) + ", " +
            indices_to_names[y_ind] + " = " + str(truncdec(event.ydata, 2)) + ", " + indices_to_names[
                remainder[0]] + " = " + str(truncdec(slider_vals[0], 2)) + ", " + indices_to_names[
                remainder[1]] + " = " + str(truncdec(slider_vals[1], 2)) + ", " + indices_to_names[
                remainder[2]] + " = " + str(truncdec(slider_vals[2], 2)) + ", " + str(
                indices_to_names[remainder[3]]) + " = " + truncdec(slider_vals[3], 2) + ",\n " + indices_to_names[
                remainder[4]] + " = " + truncdec(slider_vals[4], 2) + ", " + indices_to_names[
                remainder[5]] + " = " + str(truncdec(slider_vals[5], 2)) + ", S.D.P. = " + str(truncdec(spin_down_pops[*index_tuple], 2)) + ", Purity = " + str(truncdec(purity_data[*index_tuple], 2)) + ", neg = " + str(truncdec(numerical_negativity_original[*index_tuple], 2)))
        gs2.update(wspace=0.5)
        ax7 = plt.subplot(gs2[5:20, 5:20])
        ax8 = plt.subplot(gs2[5:20, 25:40])
        ax9 = plt.subplot(gs2[25:40, 13:28])
        fig_7 = ax7.imshow(np.imag(stored_ee_t_rho[*index_tuple]), interpolation='nearest')
        ax7.title.set_text("Imaginary Part; Energy Eigenbasis")
        fig_8 = ax8.imshow(np.imag(stored_site_t_rho[*index_tuple]), interpolation='nearest')
        ax8.title.set_text("Imaginary Part; Product Basis")
        fig_9 = ax9.imshow(np.imag(stored_site_t_pt[*index_tuple]), interpolation='nearest')
        ax9.title.set_text("Imaginary Part; Partial Transpose")
        plt.colorbar(fig_7, ax=ax7)
        plt.colorbar(fig_8, ax=ax8)
        plt.colorbar(fig_9, ax=ax9)
        plt.show()
        default_vals = slider_vals
        plot_interactive_plot(default_vals)

def plot_interactive_plot(default_vals = (np.min(indices_to_vals[remainder[0]]),np.min(indices_to_vals[remainder[1]]),np.min(indices_to_vals[remainder[2]]),np.min(indices_to_vals[remainder[3]]),np.min(indices_to_vals[remainder[4]]),np.min(indices_to_vals[remainder[5]]))):
    mpl.use('Qt5Agg')
    fig = plt.figure()
    gs = gridspec.GridSpec(40, 40)
    ax = fig.add_subplot(gs[:])
    ax.axis("off")
    sax0 = fig.add_axes([0.05, 0.05, 0.075, 0.03])
    sax1 = fig.add_axes([0.25, 0.05, 0.075, 0.03])
    sax2 = fig.add_axes([0.05, 0.15, 0.075, 0.03])
    sax3 = fig.add_axes([0.25, 0.15, 0.075, 0.03])
    sax4 = fig.add_axes([0.05, 0.25, 0.075, 0.03])
    sax5 = fig.add_axes([0.25, 0.25, 0.075, 0.03])
    slider_0 = Slider(ax=sax0, label=indices_to_names[remainder[0]], valmin=np.min(indices_to_vals[remainder[0]]), valmax=np.max(indices_to_vals[remainder[0]]), valinit=default_vals[0], orientation="horizontal", valstep = indices_to_vals[remainder[0]])
    slider_1 = Slider(ax=sax1, label=indices_to_names[remainder[1]], valmin=np.min(indices_to_vals[remainder[1]]), valmax=np.max(indices_to_vals[remainder[1]]), valinit=default_vals[1], orientation="horizontal", valstep = indices_to_vals[remainder[1]])
    slider_2 = Slider(ax=sax2, label=indices_to_names[remainder[2]], valmin=np.min(indices_to_vals[remainder[2]]), valmax=np.max(indices_to_vals[remainder[2]]), valinit=default_vals[2], orientation="horizontal", valstep = indices_to_vals[remainder[2]])
    slider_3 = Slider(ax=sax3, label=indices_to_names[remainder[3]], valmin=np.min(indices_to_vals[remainder[3]]), valmax=np.max(indices_to_vals[remainder[3]]), valinit=default_vals[3], orientation="horizontal", valstep = indices_to_vals[remainder[3]])
    slider_4 = Slider(ax=sax4, label=indices_to_names[remainder[4]], valmin=np.min(indices_to_vals[remainder[4]]), valmax=np.max(indices_to_vals[remainder[4]]), valinit=default_vals[4], orientation="horizontal", valstep = indices_to_vals[remainder[4]])
    slider_5 = Slider(ax=sax5, label=indices_to_names[remainder[5]], valmin=np.min(indices_to_vals[remainder[5]]), valmax=np.max(indices_to_vals[remainder[5]]), valinit=default_vals[5], orientation="horizontal", valstep = indices_to_vals[remainder[5]])

    def update(val):
        ind_0 = list(indices_to_vals[remainder[0]]).index(slider_0.val)
        ind_1 = list(indices_to_vals[remainder[1]]).index(slider_1.val)
        ind_2 = list(indices_to_vals[remainder[2]]).index(slider_2.val)
        ind_3 = list(indices_to_vals[remainder[3]]).index(slider_3.val)
        ind_4 = list(indices_to_vals[remainder[4]]).index(slider_4.val)
        ind_5 = list(indices_to_vals[remainder[5]]).index(slider_5.val)
        inds = (ind_0, ind_1, ind_2, ind_3, ind_4, ind_5)
        fig_1.set_array(spin_down_pops[:, :, *inds])
        fig_2.set_array(purity_data[:, :, *inds])
        fig_3.set_array(numerical_negativity_data[:, :, *inds])
        plt.show()

    slider_0.on_changed(update)
    slider_1.on_changed(update)
    slider_2.on_changed(update)
    slider_3.on_changed(update)
    slider_4.on_changed(update)
    slider_5.on_changed(update)
    gs.update(wspace=0.5)
    ax1 = plt.subplot(gs[5:20, 5:20])
    ax2 = plt.subplot(gs[5:20, 25:40])
    ax3 = plt.subplot(gs[25:40, 25:40])
    default_indices = (list(indices_to_vals[remainder[0]]).index(default_vals[0]), list(indices_to_vals[remainder[1]]).index(default_vals[1]), list(indices_to_vals[remainder[2]]).index(default_vals[2]), list(indices_to_vals[remainder[3]]).index(default_vals[3]), list(indices_to_vals[remainder[4]]).index(default_vals[4]), list(indices_to_vals[remainder[5]]).index(default_vals[5]))
    pop_min = np.min(spin_down_pops)
    pop_max = np.max(spin_down_pops)
    fig_1 = ax1.pcolormesh(indices_to_vals[x_ind], indices_to_vals[y_ind], spin_down_pops[:,:, *default_indices].reshape(len(indices_to_vals[y_ind]), len(indices_to_vals[x_ind])))
    fig_1.set_clim(pop_min, pop_max)
    ax1.title.set_text("Spin Down Pops")
    purity_min = np.min(spin_down_pops)
    purity_max = np.max(spin_down_pops)
    fig_2 = ax2.pcolormesh(indices_to_vals[x_ind], indices_to_vals[y_ind], purity_data[:,:, *default_indices].reshape(len(indices_to_vals[y_ind]), len(indices_to_vals[x_ind])))
    fig_2.set_clim(purity_min, purity_max)
    ax2.title.set_text("Purity")
    cm = plt.get_cmap("viridis")
    cm.set_bad("black")
    neg_min = np.min(numerical_negativity_original)
    neg_max = np.max(numerical_negativity_original)
    fig_3 = ax3.pcolormesh(indices_to_vals[x_ind], indices_to_vals[y_ind], numerical_negativity_data[:,:, *default_indices].reshape(len(indices_to_vals[y_ind]), len(indices_to_vals[x_ind])), cmap = cm)
    fig_3.set_clim(neg_min, neg_max)
    ax3.title.set_text(" Negativity (Black means neg = 0)")
    plt.colorbar(fig_1, ax=ax1)
    plt.colorbar(fig_2, ax = ax2)
    plt.colorbar(fig_3, ax = ax3)
    fig.canvas.callbacks.connect('button_press_event', lambda event : on_pick(event, (slider_0.val, slider_1.val, slider_2.val, slider_3.val, slider_4.val, slider_5.val), (fig_1.axes, fig_2.axes, fig_3.axes)))
    plt.show()

plot_interactive_plot()