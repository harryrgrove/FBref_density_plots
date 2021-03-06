import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import cm
import matplotlib.patches as patches
from celluloid import Camera

all_data = pd.concat([pd.read_csv('1920_Outfield.csv', sep='\t'),
                      pd.read_csv('1819_Outfield.csv', sep='\t'),
                      pd.read_csv('1718_Outfield.csv', sep='\t')])


# Modify dataframe to include useful per90 stats
def add_data(df):
    df['pressures_per90'] = df['pressures'] / df['minutes'] * 90
    df['passes_progressive_distance_per90'] = df['passes_progressive_distance'] / df['minutes'] * 90
    df['pressures_att_3rd_per90'] = df['pressures_att_3rd'] / df['minutes'] * 90
    df['normalised_finishing'] = (df['goals'] - df['pens_made'] + 50) / (df['npxg'] + 50)
    df['carry_progressive_distance_per90'] = df['carry_progressive_distance'] / df['minutes'] * 90
    df['nutmegs_per90'] = df['nutmegs'] / df['minutes'] * 90
    df['players_dribbled_past_per90'] = df['players_dribbled_past'] / df['minutes'] * 90
    df['touches_att_pen_area_per90'] = df['touches_att_pen_area'] / df['minutes'] * 90
    df['aerials_won_per90'] = df['aerials_won'] / df['minutes'] * 90
    df['sca_passes_per90'] = (df['sca_passes_live'] + df['sca_passes_dead']) / df['minutes'] * 90
    df['sca_dribbles_per90'] = df['sca_dribbles'] / df['minutes'] * 90
    df['touches_att_pen_area_per90'] = df['touches_att_pen_area'] / df['minutes'] * 90
    return df


# Modify FBref stats
all_data = add_data(all_data)


def get_index(player_name, season):  # Find index of player in season df from unique name substring
    # Build pandas df containing stats from given season
    season_df = pd.read_csv(str(season)[-2:] + str(int(str(season)[-2:]) + 1) + '_Outfield.csv', sep='\t', index_col=0)
    l = len(season_df[season_df['player'].str.contains(player_name)].index.values)  # No. times substring is found
    if l == 1:
        return season_df[season_df['player'].str.contains(player_name)].index.values[0]  # Return player index if unique
    else:
        raise Exception('get_index returned {} values. Needs to return 1.'.format(l))


# Draw density plot on axis, given player, stat, season, colour scheme
def densityplot(ax, players_index, seasons, prop=0, df=all_data, stat='shots_total_per90', pos='FW', cmap='RdYlBu'):
    # Build pandas df containing stats from given season
    season_df = pd.read_csv(str(seasons[0])[-2:] + str(int(str(seasons[0])[-2:]) + 1) + '_Outfield.csv', sep='\t',
                            index_col=0)
    season_df = add_data(season_df)  #  Modify stats
    val1 = season_df.loc[players_index[0]][stat]  #  Get relevant player's value

    season_df = pd.read_csv(str(seasons[1])[-2:] + str(int(str(seasons[1])[-2:]) + 1) + '_Outfield.csv', sep='\t',
                            index_col=0)
    season_df = add_data(season_df)  #  Modify stats
    val2 = season_df.loc[players_index[1]][stat]  #  Get relevant player's value

    val = val1 + (val2 - val1) * prop

    if pos == 'FW':  #  Isolate sample of forwards and wingers
        df = df[(df['position'].isin(('FW', 'FW,MF'))) & (df['npxg'] > 3) & (df['minutes'] > 500)]

    # Plot distribution on axis
    sns.kdeplot(x=stat, data=df, bw_adjust=.5, linewidth=0, ax=ax)
    l1 = ax.lines[0]
    x1, y1 = l1.get_xydata()[:, 0], l1.get_xydata()[:, 1]

    # Set axis bounds
    xl = min(x1)
    xh = max(x1)
    yl = 0
    yh = max(y1) * 1.05
    arr = [[0, 1], [0, 1]]
    q1 = df[stat].quantile(.1)
    q3 = df[stat].quantile(.9)
    min_grad, max_grad = q1, q3  # Color gradient goes over central 80% range
    cmap = cm.get_cmap(cmap)

    # Draw gradient between bounds
    im1 = ax.imshow(arr, cmap=cmap, extent=[min_grad, max_grad, yl, yh], interpolation="bicubic", alpha=1,
                    aspect="auto")

    # Draw solid block colors to the left and right of gradient
    ax.add_patch(
        patches.Rectangle(
            xy=(xl, yl),
            width=min_grad - xl,
            height=yh - yl,
            linewidth=0,
            color=cmap(1 - 0.96),
            fill=True
        ),
    )
    ax.add_patch(
        patches.Rectangle(
            xy=(max_grad, yl),
            width=xh - max_grad,
            height=yh - yl,
            linewidth=0,
            color=cmap(0.962),
            fill=True
        ),
    )

    x2 = [x for x in x1 if x <= val]
    y2 = y1[:len(x2)]
    x3, y3 = x1[len(x2) - 1:], y1[len(x2) - 1:]

    #  Draw gray block through the right of val
    ax.add_patch(
        patches.Rectangle(
            xy=(x3[0], yl),
            width=x3[-1] - x3[0],
            height=yh - yl,
            linewidth=0,
            color='lightgray',
            fill=True
        ),
    )

    # Fill area above curve with white
    ax.fill_between([xl, min(x1)], [0, 0], yh + 0.1, color="white", alpha=1, zorder=5)
    ax.fill_between(x1, y1, yh + 0.1, color="white", alpha=1, zorder=5)

    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Only show ticks for min and max values
    ax.set_xticks([float(str(df[stat].quantile(.001))[:4]), float(str(df[stat].quantile(.999))[:4])])

    # Align label so it does not overlap with curve
    mode = x1[list(y1).index(max(y1))]
    ax.annotate(str(val)[:4], (val, y2[-1] + yh * 0.03), zorder=6,
                ha='right' * int(val < mode) + 'left' * int(val > mode), color='gray', fontsize=8)
    # Adjust graphical formatting components
    ax.tick_params(axis='x', colors='gray')
    ax.xaxis.labelpad = -12
    ax.set_xlim(float(str(df[stat].quantile(.001))[:4]), float(str(df[stat].quantile(.999))[:4]))
    ax.yaxis.set_visible(False)


def animate_profile(players=('Lionel Messi', 'Lookman'),
                    years=(2020, 2020),
                    stats=('npxg_per90', 'npxg_per_shot', 'xa_per90', 'sca_passes_per90', 'sca_dribbles_per90'),
                    stat_names=(
                    'Non-Penalty xG', 'NPxG per Shot', 'xA', 'Shot-Creating Passes', 'Shot-Creating Dribbles'),
                    pos='FW',
                    cmap='RdYlBu',
                    transition_frames=30,
                    fps=30):
    """
    :param players: list containing 2 players
    :param years: list containing 2 years corresponding to player
    :param stats: list containing >1 stats
    :param stat_names: list containing corresponding stat names
    :param pos: position for comparison (fwds only currently supported)
    :param cmap: colormap
    :param transition_frames: number of frames in transiition between visualisations
    :param fps: fpl
    """
    players = [get_index(players[i], years[i]) for i in range(len(players))]
    fig, ax = plt.subplots(len(stats))  # Create axes
    camera = Camera(fig)
    fig.set_size_inches(4, 5 / 4 * len(stats))
    fig.text(1, 0, '@ff_trout     data: FBref/StatsBomb',  #  Add watermark
             fontsize=10, color='gray',
             ha='right', va='bottom', alpha=0.3)
    for i, prop in enumerate((np.cos(np.concatenate(
            [np.linspace(0, 1, transition_frames), [1] * fps, np.linspace(1, 2, transition_frames),
             [2] * fps]) * np.pi) + 1) / 2): # Iterates prop from 1 to 0 to 1, smoothed using a trig mapping
        print('{}/{} frames rendered'.format(i, transition_frames * 2 + fps * 2))

        # Add dynamic title
        if prop <= 0.5:
            ax[0].text(0.5, 1,
                       pd.read_csv(str(years[0])[-2:] + str(int(str(years[0])[-2:]) + 1) + '_Outfield.csv', sep='\t',
                                   index_col=0).loc[players[0]]['player'] + ' {}/{}'.format(str(years[0])[-2:], str(
                           int(str(years[0])[-2:]) + 1)), color=np.array([0, 0, 0]) + prop * 2, zorder=6,
                       transform=ax[0].transAxes, ha='center', va='bottom')
        if prop > 0.5:
            ax[0].text(0.5, 1,
                       pd.read_csv(str(years[1])[-2:] + str(int(str(years[1])[-2:]) + 1) + '_Outfield.csv', sep='\t',
                                   index_col=0).loc[players[1]]['player'] + ' {}/{}'.format(str(years[1])[-2:], str(
                           int(str(years[1])[-2:]) + 1)), color=np.array([0, 0, 0]) + (1 - prop) * 2, zorder=6,
                       transform=ax[0].transAxes, ha='center', va='bottom')

        for i, stat in enumerate(stats):
            densityplot(ax[i], players_index=players, seasons=years, prop=prop, stat=stat, cmap=cmap,
                        pos=pos)  # Density plots

        for i in range(len(stats)):
            ax[i].set_xlabel(stat_names[i])  #  Label plots
        plt.tight_layout()
        camera.snap()
    ani = camera.animate()
    ani.save('animation.gif', fps=fps) # Save animation


animate_profile(players=('Grealish', 'Grealish'),
                years=(2019, 2020),
                stats=('npxg_per90', 'npxg_per_shot', 'xa_per90', 'sca_passes_per90', 'sca_dribbles_per90'),
                stat_names=('Non-Penalty xG', 'NPxG per Shot', 'xA', 'Shot-Creating Passes', 'Shot-Creating Dribbles'),
                pos='FW',
                cmap='RdYlBu',
                transition_frames=50,
                fps=30)
