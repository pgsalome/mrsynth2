import random
import torch
from typing import List, Optional


class ImagePool:
    """This class implements an image buffer that stores previously generated images.

    This buffer enables dynamic update of discriminator with a history of generated images
    rather than only the ones produced by the latest generators.
    """

    def __init__(self, pool_size: int = 50):
        """Initialize the ImagePool class.

        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images: List[torch.Tensor] = []

    def query(self, images: torch.Tensor) -> torch.Tensor:
        """Return an image from the pool.

        Parameters:
            images (torch.Tensor) -- the latest generated images from the generator

        Returns:
            A batch of images from the buffer, with some probability the returned
            images are replaced by the latest generated ones
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images

        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)

            if self.num_imgs < self.pool_size:
                # If the buffer is not full; keep inserting current images
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                # If the buffer is full
                p = random.uniform(0, 1)
                if p > 0.5:  # With 50% probability, we replace an old image with a new image
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:  # With 50% probability, we return a previously stored image
                    random_id = random.randint(0, self.pool_size - 1)
                    return_images.append(self.images[random_id].clone())

        # Concatenate all the images in the return_images list
        return torch.cat(return_images, 0)