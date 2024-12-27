import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import Image as IPImage
from io import BytesIO
from beeoptimal import ArtificialBeeColony


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_data = datasets.FashionMNIST(root='../docs/source/_static/MNISTdata', train=True, download=True, transform=transform)


image_index = 9
target_image = train_data[image_index][0].flatten().numpy()
fig, ax = plt.subplots(figsize=(6, 4))
cax = ax.imshow(target_image.reshape(28,28), cmap='Blues')
fig.colorbar(cax)
fig.tight_layout()

def reconstruction_error(x,target=target_image):
    """_summary_
    Args:
        x (numpy array)     : vector encoding a candidate image
        target (numpy array): vector encoding the target image
    Returns:
        float: the reconstruction error
    """
    return np.sum((x - target) ** 2)


MUTATIONS = ['StandardABC','ModifiedABC','ABC/best/1','DirectedABC']
MUTATION_NAMES = ['StandardABC','ModifiedABC','ABCbest1','DirectedABC']
CMAP = ['Blues','Greens','Reds','Purples']

for idx,mutation in enumerate(MUTATIONS):
    ABC = ArtificialBeeColony(
    colony_size = 100,
    function    = reconstruction_error,
    bounds      = np.array([(0.0, 1.0)] * 784)
    )
    ABC.optimize(verbose=True,mutation=mutation,max_iters=4000,mr=0.5)

    plots = []
    # Adaptive step in order to have gifs with same number of frames
    step = max(1, (ABC.actual_iters + 1) // 50)  # Note: actual_iters +1 to include initial population
    # Generate plotly frames
    for iteration in range(0, (ABC.actual_iters +step), step):  # Note: actual_iters +step be sure including the final population
        fig, ax = plt.subplots(figsize=(6,4))
        cax = ax.imshow(ABC.optimal_bee_history[iteration].position.reshape(28, 28), cmap=CMAP[idx])
        fig.colorbar(cax)
        ax.set_title(f"Iteration {iteration}")
        fig.tight_layout()
        plt.close(fig)
        plots.append(fig)
        
    images = []
    for fig in plots:
        # Save each figure to a BytesIO object in memory instead of a file
        img_buf = BytesIO()
        fig.savefig(img_buf, format="png",dpi=100)       # Matplotlib version
        img_buf.seek(0)  # Rewind the buffer to the start
        images.append(Image.open(img_buf))  # Open the image from the buffer
        
    # Create the GIF
    gif_path = f"images/image_reconstruction/reconstruction_{MUTATION_NAMES[idx]}.gif"
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=200, loop=0)