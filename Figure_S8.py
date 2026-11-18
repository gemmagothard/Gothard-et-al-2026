import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from code_utils import analysis_plotting as pl
plt.rcParams.update({
    'font.family': 'Arial',
    'svg.fonttype': 'none',
    'font.size':10
})

colors = {
        'OP-der':'#00CDE5',
        'IP-der': '#CB23E0',
        }


PV_csv = pd.read_csv('/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_2026/Source_data/Figure 6/PV_input_results.csv')
PV_csv = PV_csv[PV_csv.id=='pv_inputs']
starter_csv = pd.read_csv('/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_2026/Source_data/Figure 6/starter_cell_results.csv')

PV_csv = PV_csv[PV_csv.condition.isin(colors.keys())]
starter_csv  = starter_csv[starter_csv.condition.isin(colors.keys())]

print(starter_csv.groupby(['condition']).count())


fig = plt.figure(figsize=(9.27, 11.69/4))
gs = fig.add_gridspec(1, 3, hspace=0.4, wspace=0.3)
# Row 1: 3 panels
ax_spread = fig.add_subplot(gs[0,0])
ax_pial = fig.add_subplot(gs[0,1])
ax_ratio = fig.add_subplot(gs[0,2])
pl.add_panel_letter(fig, gs[0, 0], "a")
pl.add_panel_letter(fig, gs[0, 1], "b")
pl.add_panel_letter(fig, gs[0, 2], "c")


dv = 'starter_dist_to_center'
iv = 'condition'
sns.barplot(starter_csv,
             x=iv,
             y=dv,
             hue=iv,
             palette=colors,
             errorbar='se',
             ax=ax_spread,
             alpha=.5)
sns.stripplot(starter_csv,
             x=iv,
             y=dv,
             palette=colors,
             hue=iv,
             ax=ax_spread)
ax_spread.set_ylabel('Starter neuron spread (\u03BCm)')
ax_spread.set_xlabel('')
categories = starter_csv[iv].unique()
t,p,posthoc  = pl.perform_stat_test(starter_csv, dv, iv)
print(t)
print(posthoc)

dv = 'Y'
iv = 'condition'
sns.barplot(starter_csv,
             x=iv,
             y=dv,
             hue=iv,
             palette=colors,
             errorbar='se',
             ax=ax_pial,
             alpha=.5)
sns.stripplot(starter_csv,
             x=iv,
             y=dv,
             palette=colors,
             hue=iv,
             ax=ax_pial)
ax_pial.set_ylabel('Starter cell pial distance (\u03BCm)')
ax_pial.set_xlabel('')
categories = starter_csv[iv].unique()
t,p,posthoc = pl.perform_stat_test(starter_csv, dv, iv)
print(t)
print(posthoc)



PV_n = PV_csv.groupby(['condition','brain_ID'])['pv_id'].count()
starter_n = starter_csv.groupby(['condition','brain_ID'])['starter_id'].count()
pv_per_starters = (
    (PV_n / starter_n)
    .rename('PV_per_starter')
    .reset_index()
)
sns.barplot(pv_per_starters,
            x='condition',
            y='PV_per_starter',
            hue='condition',
            palette=colors,
            order=list(colors.keys()),
            errorbar='se',
            alpha=.5,
            ax=ax_ratio)
sns.stripplot(pv_per_starters,
            x='condition',
            y='PV_per_starter',
            hue='condition',
            palette=colors,
            order=list(colors.keys()),
            ax=ax_ratio)
ax_ratio.set_ylabel('PVs per starter neuron')
ax_ratio.set_xlabel('')
t,p,posthoc = pl.perform_stat_test(pv_per_starters, 'PV_per_starter', iv)
print(t)
print(posthoc)
sns.despine(fig,top=True,right=True)

fig.savefig('/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_2026/Figure 6/figure_S7.svg',format='svg',bbox_inches=None)
fig.savefig('/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_2026/Figure 6/figure_S7.png')
fig.savefig('/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/DPhil/Papers/Inhibitory_subnetwork_2026/Figure 6/figure_S7.tiff',dpi=600)
plt.show()