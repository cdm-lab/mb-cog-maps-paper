import src.utils.utils as utils
import src.utils.dataset_utils as ds_utils
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

mpl.rcParams['figure.dpi'] = 200


def set_palettes():
    w4_palette = ['#9e9ac8', '#756bb1', '#fb6a4a', '#de2d26']
    talking_heads = ['#6095ca', '#fbb91e', '#6aaf75']
    # mem_palette = ['#e6508b', '#70AF41']
    mem_palette = ['#a7a9ab','#666a70']
    return w4_palette, talking_heads, mem_palette


def make_figure_2(slider_dict, output_path):
    vc_sub = 'AAURS65ZO4SXC'
    dr_sub = 'ATKNH6OR2G11O'
    ir_sub = 'AUFUUD4WG9CVO'
    model_mats = utils.create_model_matrices()
    empty_label = [''] * 10
    fontsize = 16
    fig, axs = plt.subplots(2, 3, figsize=(10, 6))
    axs[0, 0].matshow(model_mats[0], interpolation='nearest')
    axs[0, 0].set_title("Visual cooccurrence", fontsize=fontsize)

    axs[0, 1].matshow(model_mats[2], interpolation='nearest')
    axs[0, 1].set_title('Direct item association', fontsize=fontsize)

    axs[0, 2].matshow(model_mats[3], interpolation='nearest')
    axs[0, 2].set_title("Indirect item association", fontsize=fontsize)

    # Setting subject stuff
    vc_post_minus_pre = slider_dict[vc_sub][1][1] - slider_dict[vc_sub][1][0]
    np.fill_diagonal(vc_post_minus_pre, 100)
    axs[1, 0].matshow(vc_post_minus_pre, interpolation='nearest')

    axs[1, 0].set_title("Participant 1", fontsize=fontsize)

    dr_post_minus_pre = slider_dict[dr_sub][1][1] - slider_dict[dr_sub][1][0]
    np.fill_diagonal(dr_post_minus_pre, 100)
    axs[1, 1].matshow(dr_post_minus_pre, interpolation='nearest')

    axs[1, 1].set_title("Participant 2", fontsize=fontsize)

    ir_post_minus_pre = slider_dict[ir_sub][1][1] - slider_dict[ir_sub][1][0]
    np.fill_diagonal(ir_post_minus_pre, 100)
    axs[1, 2].matshow(ir_post_minus_pre, interpolation='nearest')

    axs[1, 2].set_title("Participant 3", fontsize=fontsize)
    for i in [0, 1]:
        for j in [0, 1, 2]:
            axs[i, j].tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,
                left=False,  # ticks along the top edge are off
                labelbottom=False)
            axs[i, j].set_yticklabels(empty_label)
            axs[i, j].set_xticklabels(empty_label)

    # axs[0, 2].yaxis.set_label_position("right")
    axs[0, 0].set_ylabel('Model matrices', fontsize=fontsize, labelpad=15)
    # axs[1, 2].yaxis.set_label_position("right")
    axs[1, 0].set_ylabel('Example participants', fontsize=fontsize, labelpad=15)
    filename = os.path.join(output_path, 'Fig2.png')
    fig.tight_layout()
    fig.savefig(filename, dpi=250, transparent=True)


def make_figure_3(pca_df, output_path):
    lower_indices = np.tril_indices(10, -1, 10)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 16))
    fontsize = 24
    # fig.suptitle('pre versus post, grouped by second stage')
    norm_mat = np.zeros((10, 10))
    norm_mat[lower_indices] = pca_df['pc_1'].values
    norm_mat = np.tril(norm_mat) + np.triu(norm_mat.T, 1)

    norm_mat2 = np.zeros((10, 10))
    norm_mat2[lower_indices] = pca_df['pc_2'].values
    norm_mat2 = np.tril(norm_mat2) + np.triu(norm_mat2.T, 1)

    norm_mat3 = np.zeros((10, 10))
    norm_mat3[lower_indices] = pca_df['pc_3'].values
    norm_mat3 = np.tril(norm_mat3) + np.triu(norm_mat3.T, 1)

    cax = ax1.matshow(norm_mat, interpolation='nearest')
    ax1.set_title("Principal component 1", fontsize=fontsize)
    cax2 = ax2.matshow(norm_mat2, interpolation='nearest')
    ax2.set_title("Principal component 2", fontsize=fontsize)
    cax3 = ax3.matshow(norm_mat3, interpolation='nearest')
    ax3.set_title("Principal component 3", fontsize=fontsize);
    fig.set_facecolor('w');
    cb1 = fig.colorbar(cax, ax=ax1, fraction=0.046)
    cb2 = fig.colorbar(cax2, ax=ax2, fraction=0.046)
    cb3 = fig.colorbar(cax3, ax=ax3, fraction=0.046)
    for i in [cb1, cb2, cb3]:
        for t in i.ax.get_yticklabels():
            t.set_fontsize(fontsize)
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    fig.tight_layout()
    filename_fig_3 = os.path.join(output_path, 'Fig3.png')
    fig.savefig(filename_fig_3, dpi=200, bbox_inches="tight")


def make_figure_4(model_mat_fits, melted_mmf, output_path,exp):
    y, talking_heads, _ = set_palettes()
    fontsize = 14
    height = 5.4
    model_melt = model_mat_fits.melt(id_vars=['subid', 'high_arm'], var_name='state', value_name='coef')
    # making figure 4A
    g = sns.catplot(data=model_melt, x='state', y='coef', kind='strip', height=height, palette=talking_heads)
    ax = sns.pointplot(data=model_melt, y='coef', x='state', color='black', join=False, ci=95, scale=0.7)
    custom_label = ['Visual cooccurrence', 'Direct item association', 'Indirect item association']
    # custom_label = ['', '', '']
    plt.title('behRSA model matrix fits', y=1.1, fontsize=fontsize)
    # adding zero line
    plt.setp(ax.lines, zorder=100)
    plt.setp(ax.collections, zorder=100, label="")

    plt.axhline(0, linestyle='--', color='red', zorder=120)
    # plotting significance asterisks
    plt.text(0,ax.get_ylim()[1],'*',ha='center',color='black',fontsize=fontsize+6,weight='bold')
    plt.text(1,ax.get_ylim()[1],'*',ha='center',color='black',fontsize=fontsize+6,weight='bold')
    plt.text(2,ax.get_ylim()[1],'*',ha='center',color='black',fontsize=fontsize+6,weight='bold')
    g.set_xticklabels(custom_label, rotation=30, fontsize=fontsize)
    round_ticks = [int(x) for x in g.axes[0, 0].get_yticks()]
    g.set_yticklabels(round_ticks, fontsize=fontsize)
    g.set_ylabels(r'Model matrix fit ($\bar{\beta}$)', fontsize=fontsize)
    g.set_xlabels('')
    g.fig.set_facecolor('w')
    filename_fig_4a = os.path.join(output_path, 'Fig4A.png')
    g.fig.text(0,1.05,'A',fontsize=fontsize+10,weight='bold')
    g.savefig(filename_fig_4a, facecolor='w', dpi=300)
    # making figure 4B
    g = sns.catplot(data=melted_mmf, x='coef', y='value', kind='strip', height=height, palette=talking_heads)
    ax = sns.pointplot(data=melted_mmf, x='coef', y='value', color='black', join=False, ci=95, scale=0.7)
    plt.title('Difference score between high-reward and low-reward', y=1.1, fontsize=fontsize)
    # adding zero line
    plt.setp(ax.lines, zorder=100)
    plt.setp(ax.collections, zorder=100, label="")
    plt.axhline(0, linestyle='--', color='red', zorder=120)
    # plotting significance asterisk
    if exp == 2:
        plt.text(2,93,'*',ha='center',color='black',fontsize=fontsize+6,weight='bold')
    g.set_xticklabels(custom_label, rotation=30, fontsize=fontsize)
    round_ticks = [int(x) for x in g.axes[0, 0].get_yticks()]
    g.set_yticklabels(round_ticks, fontsize=fontsize)
    g.set_ylabels(r'High stake context ($\bar{\beta}$) - Low stake context ($\bar{\beta}$)', fontsize=fontsize)
    g.set_xlabels('')
    g.fig.set_facecolor('w')
    g.fig.text(-0.01,1.05,'B',fontsize=fontsize+10,weight='bold')
    filename_fig_4b = os.path.join(output_path, 'Fig4B.png')
    g.savefig(filename_fig_4b, facecolor='w', dpi=300)


def make_figure_5(model_mat_fits, grouped_stakes, w1_map_df, output_path,exp):
    fontsize = 14
    _, talking_heads, _ = set_palettes()
    mpl.rcParams['figure.figsize'] = 4.61, 8.11
    fig_5_merge = pd.merge(model_mat_fits, grouped_stakes, on='subid')
    fig_5_merge = pd.merge(fig_5_merge, w1_map_df[['subid', 'w low stakes low arm fit']], on='subid')
    fig_5_merge = fig_5_merge.rename(columns={'w low stakes low arm fit': 'w fit'})
    melted_fig_5 = fig_5_merge[
        ['subid', 'Points earned in decision-making task', 'w fit', 'Visual cooccurrence', 'Direct item association',
         'Indirect item association']].melt(id_vars=['subid', 'w fit', 'Points earned in decision-making task'],
                                            var_name='Grouping Model', value_name='Model Matrix Fit')

    melted_fig_5_round_2 = melted_fig_5.melt(id_vars=['subid', 'Grouping Model', 'Model Matrix Fit'], var_name='y_var',
                                             value_name='y_val')


    g = sns.lmplot(data=melted_fig_5_round_2, x='Model Matrix Fit', y='y_val', hue='Grouping Model',
                   col='Grouping Model', palette=talking_heads, row='y_var', sharey='row',
                   row_order=['Points earned in decision-making task', 'w fit'],
                   facet_kws={"gridspec_kws": {"wspace": 0.0, "hspace": 0.25}})
    g.fig.set_facecolor('w')
    axes = g.axes
    axes[0, 0].set_ylim(-0.05, 0.11)
    axes[1, 0].set_ylim(0, 1.0)
    titles = ['Visual cooccurrence', 'Direct item association',
              'Indirect item association']
    for i in range(axes.shape[0]):
        axes[i, 0].set_yticks(np.round(axes[i, 0].get_yticks(), 5))
        axes[i, 0].set_yticklabels(axes[i, 0].get_yticks(), fontsize=fontsize)
        for j in range(axes.shape[1]):
            axes[1, j].set_title('')
            axes[0, j].set_title(titles[j], fontsize=fontsize,y=1.1)
            axes[1, j].set_xlabel(r'Model matrix fit ($\bar{\beta}$)', fontsize=fontsize)
            round_ticks = [int(x) for x in axes[1, j].get_xticks()]
            axes[1, j].set_xticklabels(round_ticks, fontsize=fontsize)
    axes[0, 0].set_ylabel("Points earned", fontsize=fontsize)
    axes[1, 0].set_ylabel("Model-based control parameter (w)", fontsize=fontsize)
       # adding significance 

    if exp == 2:
    # adding actual correlation values 
        axes[0,1].text(95,-0.03,'*',ha='center',color='black',fontsize=fontsize+8,weight='bold')
        axes[0,2].text(95,-0.03,'*',ha='center',color='black',fontsize=fontsize+8,weight='bold')
        axes[1,1].text(95,0.1,'*',ha='center',color='black',fontsize=fontsize+8,weight='bold')
        axes[1,2].text(95,0.1,'*',ha='center',color='black',fontsize=fontsize+8,weight='bold')
        axes[1,0].text(95,0.1,'~',ha='center',color='black',fontsize=fontsize+8,weight='bold')
        axes[0,1].text(85,-0.04,r'$r=0.32$',ha='center',color='black',fontsize=fontsize)
        axes[0,2].text(85,-0.04,r'$r=0.45$',ha='center',color='black',fontsize=fontsize)
        axes[1,0].text(85,0.04,r'$r=-0.14$',ha='center',color='black',fontsize=fontsize)
        axes[1,1].text(85,0.04,r'$r=0.32$',ha='center',color='black',fontsize=fontsize)
        axes[1,2].text(85,0.04,r'$r=0.46$',ha='center',color='black',fontsize=fontsize)
    else:
        axes[0,1].text(80,-0.04,r'$r=0.34$',ha='center',color='black',fontsize=fontsize)
        axes[0,2].text(80,-0.04,r'$r=0.36$',ha='center',color='black',fontsize=fontsize)
        axes[1,1].text(80,0.04,r'$r=0.27$',ha='center',color='black',fontsize=fontsize)
        axes[1,2].text(80,0.04,r'$r=0.25$',ha='center',color='black',fontsize=fontsize)
        axes[0,1].text(85,-0.03,'*',ha='center',color='black',fontsize=fontsize+8,weight='bold')
        axes[0,2].text(85,-0.03,'*',ha='center',color='black',fontsize=fontsize+8,weight='bold')
        axes[1,1].text(85,0.1,'*',ha='center',color='black',fontsize=fontsize+8,weight='bold')
        axes[1,2].text(85,0.1,'*',ha='center',color='black',fontsize=fontsize+8,weight='bold')
    g.fig.text(0.05,0.90,'A',fontsize=fontsize+10,color='black',weight='bold')
    g.fig.text(0.05,0.45,'B',fontsize=fontsize+10,color='black',weight='bold')
    g.savefig(os.path.join(output_path, 'Fig5.png'), dpi=300)


def make_figure_6(dprime_df, output_path):
    _, _, mem = set_palettes()
    stat_dprime_df = dprime_df.groupby(['subid']).first().reset_index()
    stat_dprime_df['lbst_diff_score'] = stat_dprime_df['dprime_lbst_high'] - stat_dprime_df['dprime_lbst_low']
    stat_dprime_df['lt_diff_score'] = stat_dprime_df['dprime_lt_high'] - stat_dprime_df['dprime_lt_low']
    melted_dprime_df = stat_dprime_df[['subid', 'lbst_diff_score', 'lt_diff_score']].melt(id_vars='subid',
                                                                                          var_name='type of trial',
                                                                                          value_name="d' difference score")

    fontsize = 11
    g = sns.catplot(data=melted_dprime_df, x='type of trial', y="d' difference score", height=5, palette=mem)
    ax = sns.pointplot(data=melted_dprime_df, x='type of trial', y="d' difference score", join=False, color='Black',
                       scale=0.7)

    plt.setp(ax.lines, zorder=100)
    plt.setp(ax.collections, zorder=100, label="")
    plt.axhline(0, linestyle='--', color='red', zorder=120)
    g.set_xticklabels(["Mismatch d'", "Lure d'"], fontsize=fontsize)
    g.set_xlabels('')
    g.set_ylabels("d' difference score (high vs low stake context)", fontsize=fontsize)
    g.axes[0, 0].set_yticklabels(g.axes[0, 0].get_yticks(), fontsize=fontsize)
    g.fig.set_facecolor('w')
    g.savefig(os.path.join(output_path, 'Fig6.png'), dpi=300)


def main():
    parser = argparse.ArgumentParser(description='Code to generate primary figures for behavioral paper')
    parser.add_argument('-i', '--input_path', required=True, help='point to where you are looking for files')
    parser.add_argument('-o', '--output_path', required=True, help='where do you want the figures to be saved')
    parser.add_argument('-e', '--exp', required=False, help='specify which experiment version you are running',
                        default=2, type=int)
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    exp = args.exp
    assert os.path.exists(input_path), 'provided input path does not exist!'
    assert os.path.exists(output_path), 'provided output path does not exist!'
    data = ds_utils.load_exp_data(input_path)
    slider_dict = data['slider_dict']
    model_mat_fits = data['model_mat_fits']
    melted_mmf = data['melted_mmf']
    w1_map_df = data['w1_map_df']
    grouped_stakes = data['grouped_stakes']
    if exp == 2:
        make_figure_2(slider_dict, output_path)
        pca_df = data['pca_df']
        dprime_df = data['dprime_df']
        make_figure_6(dprime_df, output_path)
    make_figure_4(model_mat_fits, melted_mmf, output_path,exp)
    make_figure_5(model_mat_fits, grouped_stakes, w1_map_df, output_path,exp)


if __name__ == '__main__':
    main()

# TODO figure out the sizing and labelling of figures with labels (i.e. A B C)
