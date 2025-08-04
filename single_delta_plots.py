import matplotlib.pyplot as plt
import numpy as np
import math

plt.rcParams.update({"text.usetex": True, "font.family": "Computer Modern"})

omega = 1
delta = 2
n_max = 3

#Figure 1

JC_data = np.load("C:/Users/akash/PycharmProjects/CSV_for_Github/CSV/Single_Del_Cross_Sections/JC_Del_2_beta_90.npy", allow_pickle=True)
Rabi_data = np.load("C:/Users/akash/PycharmProjects/CSV_for_Github/CSV/Single_Del_Cross_Sections/Rabi_Del_2_beta_90.npy", allow_pickle=True)
SymBreak1 = np.load("C:/Users/akash/PycharmProjects/CSV_for_Github/CSV/Single_Del_Cross_Sections/SymBreak_Rabi_Del_2_beta_90_eps_01Del.npy", allow_pickle=True)
SymBreak2 = np.load("C:/Users/akash/PycharmProjects/CSV_for_Github/CSV/Single_Del_Cross_Sections/SymBreak_Rabi_Del_2_beta_90_eps_035Del.npy", allow_pickle=True)

fig, ax = plt.subplots()

ax.plot(JC_data[0,:], JC_data[1,:], c = "red", linestyle = "--", linewidth = 1.5)
ax.plot(JC_data[0,:], np.sqrt(JC_data[2,:]), c = "red", alpha = 0.75)
# ax.plot(JC_data[0,:], np.square(JC_data[1,:]), c = "red", linestyle = "--")
# ax.plot(JC_data[0,:], JC_data[2,:], c = "red")


ax.plot(Rabi_data[0,:], Rabi_data[1,:], c = "blue", linestyle = "--", linewidth = 1.5)
ax.plot(Rabi_data[0,:], np.sqrt(Rabi_data[2,:]), c = "blue", alpha = 0.75)
# ax.plot(Rabi_data[0,:], np.square(Rabi_data[1,:]), c = "blue", linestyle = "--")
# ax.plot(Rabi_data[0,:], Rabi_data[2,:], c = "blue")


ax.plot(SymBreak1[0,:], SymBreak1[1,:], c = "green", linestyle = "--", linewidth = 1.5)
ax.plot(SymBreak1[0,:], np.sqrt(SymBreak1[2,:]), c = "green", alpha = 0.75)
# ax.plot(SymBreak1[0,:], np.square(SymBreak1[1,:]), c = "green", linestyle = "--")
# ax.plot(SymBreak1[0,:], SymBreak1[2,:], c = "green")


# ax.plot(SymBreak2[0,:], SymBreak2[1,:], c = "orange", linestyle = "--")
# # ax.plot(SymBreak2[0,:], SymBreak2[2,:], c = "orange")
# ax.plot(SymBreak2[0,:], np.sqrt(SymBreak2[2,:]), c = "orange")

ax.set_xlabel("$\lambda$", fontsize = 16)
plt.xlim(-0.02, 4.02)
plt.ylim(-0.0025, 0.5025)

lam0 = math.sqrt(omega*delta)
ax.scatter(lam0, 0.5025, color='black', clip_on=False)
ax.annotate("$\lambda_0$", (lam0 - 0.12, 0.5025 - 0.02))
for k in range(1, n_max + 1):
    lamk = math.sqrt(2*omega**2*(1/2 + k + math.sqrt((1/2 + k)**2 + (delta - omega)**2/(2*omega)**2)))
    ax.scatter(lamk, 0.5025, color='black', clip_on=False)
    ax.annotate(f"$\lambda_{str(k)}$", (lamk - 0.12, 0.5025 - 0.02))


# plt.savefig("fig1", bbox_inches="tight", dpi = 300)
plt.show()


#Figure 2

AllEvenData = np.load("C:/Users/akash/PycharmProjects/CSV_for_Github/CSV/Single_Del_Cross_Sections/SymBaths2to1FullRange.npy", allow_pickle=True)
FiftyFiftyData = np.load("C:/Users/akash/PycharmProjects/CSV_for_Github/CSV/Single_Del_Cross_Sections/SymBathFiftyFiftyFull.npy", allow_pickle=True)
SymBreakSymBathData = np.load("C:/Users/akash/PycharmProjects/CSV_for_Github/CSV/Single_Del_Cross_Sections/SymBathsSymBreakFullRange.npy", allow_pickle=True)


fig, ax = plt.subplots()
ax.plot(AllEvenData[0,:], AllEvenData[1,:], c = "red", linestyle = "--", linewidth = 1.5)
# ax.plot(AllEvenData[0,:], AllEvenData[2,:], c = "red")
ax.plot(AllEvenData[0,:], np.sqrt(AllEvenData[2,:]), c = "red", alpha = 0.75)
ax.plot(FiftyFiftyData[0,:], FiftyFiftyData[1,:], c = "blue", linestyle = "--", linewidth = 1.5)
# ax.plot(FiftyFiftyData[0,:], FiftyFiftyData[2,:], c = "blue")
ax.plot(FiftyFiftyData[0,:], np.sqrt(FiftyFiftyData[2,:]), c = "blue", alpha = 0.75)
ax.plot(SymBreakSymBathData[0,:], SymBreakSymBathData[1,:], c = "green", linestyle = "--", linewidth = 1.5)
# ax.plot(SymBreakSymBathData[0,:], SymBreakSymBathData[2,:], c = "green")
ax.plot(SymBreakSymBathData[0,:], np.sqrt(SymBreakSymBathData[2,:]), c = "green", alpha = 0.75)



ax.set_xlabel("$\lambda$", fontsize = 16)
plt.xlim(-0.02, 4.02)
plt.ylim(-0.0025, 0.2925)
# plt.savefig("fig2", bbox_inches="tight", dpi = 300)
plt.show()