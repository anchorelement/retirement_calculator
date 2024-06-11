import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
from datetime import datetime
import yaml


def retirement_projection(
    P: float, r: list, t: int, a: int, inf: float, M=0, series: int = 0
) -> pd.DataFrame:
    """Calculates the the principal savings amount
    P: Principal Amount
    r: list of expected annual rates of return as fractions (e.g. .06 for 6%)
    inf: expected rate of inflation as a fraction.
    t: time in years
    a: starting age
    M: additional monthly contribution/deduction
    """
    data = []
    prior_principal = P
    prior_contribution = M
    global failed_simulations
    # Update the total amount at the end of every month, taking into account contributions, returns and inflation
    for period in range(t * 12):
        # Investment return
        current_rate = r[period]
        current_principal = prior_principal * (1 + current_rate / 12)
        # Contribution
        current_contribution = prior_contribution * (1 + inf / 12)
        # Calculate new amount
        amount = current_principal + current_contribution
        # Rollover variables for next iteration
        prior_contribution = current_contribution
        prior_principal = amount
        if amount > 0:
            data.append(
                {
                    "Change": round(current_contribution, 2),
                    "Age": a + period // 12,
                    "Amount": round(amount, 2),
                    "Rate": current_rate,
                    "Series": series,
                }
            )
        else:
            failed_simulations += 1
            break
    return pd.DataFrame(
        data=data, columns=["Age", "Amount", "Change", "Rate", "Series"]
    )


# Load configuration file
with open("sample_input.yaml") as config_file:
    config = yaml.safe_load(config_file)
P = config.get("starting_principal")
M = config.get("monthly_contributions")
n = config.get("compound_frequency")
r = config.get("rate_of_return")
inf = config.get("rate_of_inflation")
num_simulations = config.get("num_simulations")
failed_simulations = 0

year = datetime.now().year
age = config.get("current_age")
retirement_age = config.get(
    "retirement_age"
)  # Assumes retirement starts on reaching provided age
life_expectancy = config.get("life_expectancy")
t1 = retirement_age - age
t2 = life_expectancy - retirement_age
y_i_r = config.get("yearly_income_in_retirement")  # Pension, Social Security, etc...
y_s_r = config.get("yearly_spend_in_retirement")
I = float(y_i_r / 12)
E = float(y_s_r / 12)

# Configure normal distribution params
loc = 0.06
scale = 0.1
size = (t1 + t2) * 12

data_frames = []

for series in range(num_simulations):
    rates = np.random.normal(loc, scale, size)
    df_pre = retirement_projection(P, rates, t1, age, inf, M, series)
    retirement_amount = df_pre.max().get("Amount")
    print(retirement_amount)

    # Calculate initial income - expenditures in retirement, adjusted for inflation
    net_income_in_retirement = (I - E) * np.power(1 + inf, t1)
    df_post = retirement_projection(
        retirement_amount,
        rates,
        t2,
        retirement_age,
        inf,
        net_income_in_retirement,
        series,
    )
    df = pd.concat([df_pre, df_post], ignore_index=True)
    data_frames.append(df)

prob_success_retirement = (
    (num_simulations - failed_simulations) / num_simulations
) * 100

data = pd.concat(data_frames, ignore_index=True)
# Dump to file for testing or export
data.to_csv("data.csv")

# Setup graph visuals and styling
pd.options.display.float_format = "{:,.2f}".format
sns.set_theme(rc={"figure.figsize": (14, 9)}, style="darkgrid")

ax = sns.lineplot(
    data=data,
    x="Age",
    y="Amount",
    errorbar=None,
    hue="Series",
    legend=False,
    palette=sns.color_palette(palette="magma", n_colors=num_simulations),
)

ax.set(
    title=f"{prob_success_retirement:.0f}% Chance of Successful Retirement",
    ylabel="Amount $",
)

ax.get_yaxis().set_major_formatter(
    mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ","))
)

plt.show()
