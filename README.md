# Solution to the Greenscreens.ai take-home assessment (Jan'24)

## Strcuture
* Task 1 Description & Solution
* Task 2 Description & Solution
* Running Instructons
* Directory Structure

## Task 1

### Description
You should predict rate per mile using your model, features etc.

Example code returns average rate as prediction, it gives 34.85% accuracy.

Rate quality prediction is measured by MAPE (mean absolute percentage error):

$$ MAPE =  \frac{1}{N}  \sum_{i=1}^{N} {\bigg | {1 - \frac{Rate_{predicted}^{i}}{Rate_{real}^{i}}}\bigg | }  \times 100\%  $$

There are three files with data: train.csv, validation.csv (validation set infront of train in term of date) and test.csv (for your prediction)
We have divided all US territory into KMA (Key Market Regions). These regions grouped by similar market conditions that are inside each market. Try to enhance the current Rate Engine by pushing knowledge about origin and destination KMA into model. 
The dataset contains the following features: the number of miles of the route, the type of the transport (there are three main types of transport), used for transporting the cargo, the weight of the cargo, the date when the cargo was picked up, the KMA origin point and the KMA destination point.

Try to beat our prediction accuracy (MAPE), we are expecting less than 9%.

Please send estimate your validation MAPE, the predictions for the test.csv and don't forget to attach your code.

### Solution

#### Data Analysis
#### Data Transformations
#### Searching the Model Space
#### Searching the Parameter Space
#### Fit and Test
#### Results
#### Future Work

## Task 2 

### Description

Think and propose your version of a asymmetric MAPE-Like Metric. This asymmetry should be adjustable through the introduction of an additional parameter, allowing for fine-tuning of the metric's sensitivity to overestimations and underestimations. Consider the advantages, disadvantages, and limitations of using such a metric.

### Solution

First, let's establish the over and under estimation conditions.

Let $Rate_{real}$ represent real rates and $Rate_{predicted}$ represent the predicted rates.
Then, the conditions could be represented as follows:

$Rate_{real} - Rate_{predicted} < 0$ for overestimation; and

$Rate_{real} - Rate_{predicted} > 0$ for underestimation.

To allow the asymetric behaviour for adjustable sensitivity to either over or under estimation, a parameter alpha ($\alpha$) could be introduced to control the weighting given to each of the cases accordingly:

$$
\begin{aligned}
    \alpha * \bigg | { {\frac{Rate_{real}-Rate_{predicted}}{Rate_{real}}}}\bigg | \quad for \: real - predicted <0\\ 
    and\quad(1 - \alpha) * \frac{Rate_{real}-Rate_{predicted}}{Rate_{real}}  \quad for \: real - predicted \geq0\\ 
\end{aligned}
$$

This can be represented in an equation as presented below in a simplified form:

$$ 
aMAPE =  \frac{1}{N}  \sum_{i=1}^{N} {|\alpha*over\:+\:(1-\alpha)*under |}  \times 100\%  
$$

This means that adjusting $\alpha$ would change the behavour of the cost function as follows:

* $\alpha$ > 0.5 increase the penalty for overestimation
* $\alpha$ < 0.5 increase the penalty for underestimation
* $\alpha$ = 0.5 force symetric behaviour

This equation has been coded up in full and can be found in `utils/loss_functions.py::amape_loss`.
A script comparing aMAPE and MAPE can be found at `t2_visualise_losses.py` as well as its output which is saved as `visualisations/losses.png`. 



This image represents a diagram as shown below, it compares MAPE and aMAPE losses behaviour depending on conditions. Samples 0-4 represnt (mostly severe) underestimation, 5-8 (balanced estimation) and 9-12 severe overestimation.

![Alt text](visualisations/losses.png)

This digram is helps us demostrate the effect that adjustment of the parameter $\alpha$ has on the behavour of the aMAPE loss fuction. We see that in the edge cases such as $\alpha$=0 or $\alpha$=1, aMAPE loss repeats the MAPE line exactly for the underestimation and overestimation regimes, respectively, while showing 0% loss for the oposite regime. This behaviour is undesirable in real cases as it prevents any analysis beyond the respective operating window. However, a well balanced $\alpha$ could help adjust the system depending on the needs of the application making the loss more or less sensitive to either over or under estimations. However, this flexibility comes at a cost of the aMAPE being harder to interpret and communicate as it is not anymore a straightforward ratio as compared to MAPE. Furthermore, the selection of $\alpha$ becomes critical as it can lead to unwanted biases.

In summary, the following list represents advantages and disadvantages of aMAPE vs MAPE:
* Advantages:
    * Flexibility
    * Business Relevance - setup your system to prefer either over or under predictions depending on the "cost to business", e.g. it may be better to have extra cash saved vs being just over the available limit
    * Shoots off less for very incorrect predictions
* Disadvantages:
    * Interpretability - may be harder to explain
    * Parameter Selection - poorly selected $\alpha$ may lead to unwanted biases

### Install cudm and other nvida libs: 

link: https://rapids.ai/#quick-start

```
pip install \
--extra-index-url=https://pypi.nvidia.com \
cudf-cu12==23.12.* \
dask-cudf-cu12==23.12.* \
cuml-cu12==23.12.* \
cugraph-cu12==23.12.*
```