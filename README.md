# Resources
  * [Repository of the laboratories](https://github.com/bardhprenkaj/ML_labs2022).
  * [Drive for the data](https://drive.google.com/drive/folders/1VeZftDc1KntjZnzwevh2V9YzXGG-6ri0?usp=share_link).
  * [T-LSTM](https://ieeexplore.ieee.org/document/8767922)
  * [PubMedBERT](https://microsoft.github.io/BLURB/models.html)

# Codici AMD
Informations about codici AMD can be found here:
  * [Download](https://aemmedi.it/download/24947/)
  * [Webpage](https://aemmedi.it/annali-amd/)

# Anatomical Therapeutic Chemical Classification System (ATC codes)
Information about ATC codes can be found here:
  * [Wikipedia](https://en.wikipedia.org/wiki/Anatomical_Therapeutic_Chemical_Classification_System)
  * [Official Website](http://www.whocc.no/)
  * [Scraped Data](https://github.com/fabkury/atcd/blob/b426860423e1d0321e042184f2c61afefbaf51d9/WHO%20ATC-DDD%202021-12-03.csv)

# Notes
If you are using `matplotlib`, on a recent version macOS (e.g. Ventura), from
the command line, it can report crushes when the window goes in and out of focus
on the `stderr`.

To avoid this you can launch Python redirecting the `stderr` to `/dev/null` i.e.
`python3 2>/dev/null`. Remember to use `matplotlib.pyplot.show()` to show the
window after creating the plot, or `matplotlib.pyplot.savefig('foo.pdf')` if you
want to save the plot instead.

As a piece of trivia the backends of `matplotlib` that have `agg` in the name
use the [Anti-Grain Geometry](https://agg.sourceforge.net/antigrain.com/)
library.
