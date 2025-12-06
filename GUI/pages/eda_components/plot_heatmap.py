from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

def plot_heatmap(df, numeric_cols, plot_frame):
    for w in plot_frame.winfo_children():
        w.destroy()

    corr = df[numeric_cols].corr()

    fig = Figure(figsize=(8, 4.5), dpi=100)
    ax = fig.add_subplot(111)

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        ax=ax,
        fmt=".2f",
        cbar=True
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    ax.tick_params(axis='x', labelsize=9)
    ax.tick_params(axis='y', labelsize=9)


    fig.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)
