
![CGEM Logo](https://github.com/jrolf/simplevec/blob/main/images/THINK_DEV.jpeg)

# Collaborative Generalized Effects Modeling (CGEM) 

## Introduction

Collaborative Generalized Effects Modeling (CGEM) is an advanced statistical modeling framework designed for complex real-world data analysis. Merging traditional statistical methods with machine learning techniques, CGEM is well-suited for interpreting intricate data relationships in various domains, including business analytics and scientific research.

## Defining Characteristics of CGEM

### Formulaic Flexibility

CGEM offers extensive flexibility in model formulation, supporting a wide range of mathematical relationships. This feature is vital for modeling the complex dynamics often present in datasets, allowing for linear, non-linear, and multiplicative relationships.

### Generalization of Effects

CGEM's broad interpretation of 'effects' encompasses everything from simple constants and linear terms to sophisticated machine learning model outputs. This generalization enables CGEM to incorporate and benefit from diverse methodologies within a unified model framework.

### Iterative Refinement and Convergence

CGEM employs an iterative process to refine and converge model terms, ensuring balanced weighting and mitigating issues like overfitting or variable dominance. This process is key to enhancing the robustness and accuracy of the model.

### Causal Coherence

CGEM emphasizes maintaining causally coherent relationships, ensuring that model outputs are not only statistically significant but also meaningful and interpretable within real-world contexts.

### Integration with Machine Learning

CGEM seamlessly incorporates machine learning models as effects, combining the predictive power of machine learning with the structural integrity and interpretability of traditional statistical models.

## Core Mechanics of CGEM

CGEM operates using an iterative algorithm, which involves defining a model, incorporating various effects, and continually refining the model.

### Example Implementation

#### Installation

To install the CGEM library:

```bash
pip install --upgrade cgem
```

To verify the installation:

```bash
pip show cgem
```

#### Generating Artificial Data

Generate artificial data to simulate a causal system:

```python
import numpy  as np
import pandas as pd
from random import choice
from cgem import *

Task = '''
Create a simple causal simulation to generate a dataset
that can be used to conduct a computational proof of CGEM.
'''

def gen_artificial_data_v1(size=10000):
    """
    Generate an artificial dataset representing a causal system.

    Parameters:
    size (int): Number of data points to generate.

    Returns:
    pandas.DataFrame: A DataFrame with the generated data.
    """
    global cats,effs
    # Generating random values for the variables
    reg_var_a = np.random.normal(10, 3, size)
    reg_var_b = np.random.normal(12, 4, size)
    reg_var_c = np.random.normal(15, 5, size)

    # Calculating the effect based on the variables
    effect_x = 20.0 + (1.0 * reg_var_a) + (1.5 * reg_var_b) + (2.0 * reg_var_c)

    # Defining categories and their corresponding effects
    cats = list("ABCDEFGHIJ")
    effs = np.around(np.linspace(0.5, 1.4, len(cats)), 2)
    cat2effect = {cat: round(eff, 4) for cat, eff in zip(cats, effs)}

    # Generating categorical variable and its effect
    cat_var_d = np.array([choice(cats) for _ in range(size)])
    cat_effect_d = np.array([cat2effect[c] for c in cat_var_d])

    # Adding a noise effect
    noise_effect = np.random.uniform(0.90, 1.10, size)

    # Calculating the target variable
    target_var_z = ((effect_x) * cat_effect_d) * noise_effect

    # Constructing the dataframe
    df = pd.DataFrame({
        'TGT_Z': target_var_z,
        'REG_A': reg_var_a,
        'REG_B': reg_var_b,
        'REG_C': reg_var_c,
        'CAT_D': cat_var_d
    })
    return df

### Generate Training and Testing Data:
DF1 = gen_artificial_data_v1(size=10000)  # TRAIN DataFrame
DF2 = gen_artificial_data_v1(size=10000)  # TEST  DataFrame
```

#### Defining the Model Parameters

Defining the structure, mechanics, and constraints of the model:

```python
### DEFINE THE MASTER EFFECTS FORMULA: 
Formula = "TGT_Z = CAT_D_EFF * LIN_REG_EFF"

### DEFINE THE TERM MODEL PARAMETERS:
tparams = {
    "CAT_D_EFF": {
        'model': "CatRegModel()",  # Categorical Regression Model
        'xvars': ['CAT_D'],        # Independent variable for this effect
        'ival' : 10,               # Initial value
    },
    "LIN_REG_EFF": {
        'model': "OLS()",          # Ordinary Least Squares Model
        'xvars': ['REG_A', 'REG_B', 'REG_C'],  # Independent variables for this effect
        'ival' : 10,               # Initial value
    }
}
```

#### Model Fitting

Instantiate a CGEM model, load the parameters, and fit the model:

```python 
model = CGEM() 
model.load_df(DF1)  
model.define_form(Formula) 
model.define_terms(tparams)  
model.fit(25); 
```

#### Model Evaluation

Evaluate model performance on the TEST set:

```python
preds = model.predict(DF2) 
actuals = DF2['TGT_Z'].values
r2 = model.calc_r2(actuals, preds) 
print('CrosVal R-Squared:', round(r2, 5))
```

## Conclusion

CGEM offers a sophisticated framework for data analysis, combining the strengths of various statistical and machine learning methodologies. Its flexibility, coupled with the ability to model complex and non-linear relationships, makes it a valuable tool for data scientists and analysts. The iterative optimization process ensures model robustness, and the emphasis on causal coherence enhances the interpretability of results. CGEM's integration of diverse effects and machine learning models positions it as a versatile tool, suitable for a wide range of applications in data-driven decision-making and advanced data science.

### Author's Note:
Thanks for reading! If you have further questions about this library, please message me at "james.rolfsen@think.dev" or connect with me on LinkedIn via https://www.linkedin.com/in/jamesrolfsen/  I am excited to see the many ways people use the CGEM framework in the future. Happy modeling!


#### [END OF DOC] 