Vaccination rate in the US
================
Matthew Kay
4/26/2021

The [New York Time vaccine
tracker](https://www.nytimes.com/interactive/2020/us/covid-19-vaccine-doses.html)
currently shows a predicted proportion of people vaccinated if
vaccination continues “at the current pace”, using a linear projection.
I thought it might be helpful to try to add some uncertainty to that
line in a way that does not assume a linear trend.

![NYT Vacciation rate](nyt_vax_rate.png)

## Libraries needed for this analysis:

``` r
library(lubridate)
library(ggdist)
library(tidybayes)
library(bsts)
library(patchwork)
library(tidyverse)
library(magrittr)
library(posterior)   # pak::pkg_install("stan-dev/posterior")
library(future)
library(furrr)

theme_set(theme_ggdist())
```

## The data

I downloaded the data on what share of the population has recieved at
least one dose from
[ourworldindata.org](https://ourworldindata.org/coronavirus/country/united-states#what-share-of-the-population-has-received-at-least-one-dose-of-the-covid-19-vaccine):

``` r
download.file(
  "https://github.com/owid/covid-19-data/raw/master/public/data/vaccinations/vaccinations.csv", 
  "share-people-vaccinated-covid.csv"
)
```

Filtering down just to the US:

``` r
df = read.csv("share-people-vaccinated-covid.csv") %>%
  filter(location == "United States") %>%
  transmute(
    date = as_date(date),
    day = as.numeric(date - min(date)),
    vax = people_vaccinated_per_hundred / 100
  )

head(df)
```

    ##         date day    vax
    ## 1 2020-12-20   0 0.0017
    ## 2 2020-12-21   1 0.0018
    ## 3 2020-12-22   2     NA
    ## 4 2020-12-23   3 0.0030
    ## 5 2020-12-24   4     NA
    ## 6 2020-12-25   5     NA

The data looks like this:

``` r
df %>%
  ggplot(aes(date, vax)) +
  geom_point(na.rm = TRUE)
```

<img src="vax_files/figure-gfm/raw_data-1.png" width="682.56" />

## NYT-like linear model

First off, we can approximate what NYT is doing using linear regression
based on the last couple of points (this doesn’t line up exactly with
their model but it seems close):

``` r
df %>%
  ggplot(aes(date, vax)) +
  geom_point(na.rm = TRUE) +
  stat_smooth(formula = y ~ x, method = lm, se = FALSE, data = . %>% filter(row_number() > n() - 8), fullrange = TRUE) +
  scale_x_date(limits = c(min(df$date), make_date(2021, 10, 31))) +
  coord_cartesian(ylim = c(0, 1)) +
  geom_vline(xintercept = make_date(2021, 10, 12), alpha = 0.2) +
  geom_hline(yintercept = 0.9, alpha = 0.2) +
  annotate("text", x = make_date(2021, 5, 28), y = 0.92, label = "90%", hjust = 0, vjust = 0, size = 3.5) +
  annotate("text", x = make_date(2021, 10, 8), y = 0.5, hjust = 1, vjust = 0.5, label = "Oct 12", size = 3.5) 
```

<img src="vax_files/figure-gfm/nyt_like-1.png" width="682.56" />

However, instead of doing this I am going to fit a time series model to
the data.

## Data prep for time series model of daily changes

Analyzing the data on the raw scale is hard because (1) we know that the
percent of vaccinated people is bounded between 0 and 1 and (2) we know
that the percent of vaccinated people is always increasing. Combining
these two facts, we could instead look at the difference in the logit of
the percent of vaccinated people:

``` r
df_diff = df %>%
  select(-date) %>%
  complete(day = min(day):max(day)) %>%
  mutate(date = min(df$date) + days(day))

df_diff %>%
  ggplot(aes(date, c(NA, diff(qlogis(vax))))) +
  geom_line() +
  geom_point() +
  labs(y = "daily increase in logit(Percent vaccinated)")
```

<img src="vax_files/figure-gfm/diff_plot-1.png" width="682.56" />

Since we know that this difference must be positive (since the number of
vaccinated people cannot decrease), we might analyse this on a log
scale. That might also stabilize the variance:

``` r
df_log_diff = df_diff %>%
  mutate(log_diff_vax = c(NA, log(diff(qlogis(vax))))) %>%
  slice(-1)

df_log_diff %>%
  ggplot(aes(date, log_diff_vax)) +
  geom_line() +
  geom_point(na.rm = TRUE) +
  labs(y = "log(daily increase in logit(Percent vaccinated))")
```

<img src="vax_files/figure-gfm/log_diff_plot-1.png" width="682.56" />

There’s some missing data here, which we’ll let the model impute (FWIW I
also tried just using linear interpolation on the raw data prior to
translating it into differences and got very similar results to the
imputation approach).

## Time series model

This seems a good point to fit a time series model. We’ll fit a Bayesian
Structural Time Series model with `bsts()`. We’ll use a semi-local
linear trend [intended for long-term
forecasting](https://www.unofficialgoogledatascience.com/2017/07/fitting-bayesian-structural-time-series.html).
There’s a clear weekly trend in the numbers, which is to be expected
given differences in vaccination rates on the weekend, so we’ll also
include a seasonal component with a 7-day period. Since we’re on a log
scale even changes by as much as 0.5 or 1 are large, we’ll use some
tight-ish-looking priors here:

``` r
fit_model = function(max_date = max(df_log_diff$date)) {
  with(filter(df_log_diff, date <= max_date), bsts(log_diff_vax, 
    state.specification = list() %>%
      AddSemilocalLinearTrend(log_diff_vax,
        level.sigma.prior = SdPrior(0.5, 1),
        slope.mean.prior = NormalPrior(0,0.5),
        initial.level.prior = NormalPrior(0,0.5),
        initial.slope.prior = NormalPrior(0,0.5),
        slope.sigma.prior = SdPrior(0.5, 1),
        slope.ar1.prior = Ar1CoefficientPrior(0, 0.5)
      ) %>%
      AddSeasonal(log_diff_vax, 7, 
        sigma.prior = SdPrior(0.5, 1)
      ),
    prior = SdPrior(0.5, 1),
    niter = 40000,
    seed = 4272021 # for reproducibility
  ))
}

m = fit_model()
```

    ## =-=-=-=-= Iteration 0 Thu May 27 01:39:04 2021
    ##  =-=-=-=-=
    ## =-=-=-=-= Iteration 4000 Thu May 27 01:39:10 2021
    ##  =-=-=-=-=
    ## =-=-=-=-= Iteration 8000 Thu May 27 01:39:16 2021
    ##  =-=-=-=-=
    ## =-=-=-=-= Iteration 12000 Thu May 27 01:39:22 2021
    ##  =-=-=-=-=
    ## =-=-=-=-= Iteration 16000 Thu May 27 01:39:28 2021
    ##  =-=-=-=-=
    ## =-=-=-=-= Iteration 20000 Thu May 27 01:39:35 2021
    ##  =-=-=-=-=
    ## =-=-=-=-= Iteration 24000 Thu May 27 01:39:41 2021
    ##  =-=-=-=-=
    ## =-=-=-=-= Iteration 28000 Thu May 27 01:39:48 2021
    ##  =-=-=-=-=
    ## =-=-=-=-= Iteration 32000 Thu May 27 01:39:54 2021
    ##  =-=-=-=-=
    ## =-=-=-=-= Iteration 36000 Thu May 27 01:39:59 2021
    ##  =-=-=-=-=

Some diagnostics:

``` r
draws = as_draws(do.call(cbind, m[startsWith(names(m), "trend") | startsWith(names(m), "sigma")]))
summary(draws, median, mad, quantile2, default_convergence_measures())
```

    ## # A tibble: 6 x 8
    ##   variable                median    mad      q5      q95  rhat ess_bulk ess_tail
    ##   <chr>                    <dbl>  <dbl>   <dbl>    <dbl> <dbl>    <dbl>    <dbl>
    ## 1 sigma.obs               0.125  0.0160  0.102   0.155    1.00    4805.   10768.
    ## 2 trend.level.sd          0.122  0.0158  0.0992  0.151    1.00    4052.    8332.
    ## 3 trend.slope.mean       -0.0180 0.0133 -0.0400  0.00392  1.00   10531.   19778.
    ## 4 trend.slope.ar.coeffi~ -0.379  0.181  -0.643  -0.0669   1.00    3913.    8190.
    ## 5 trend.slope.sd          0.146  0.0196  0.118   0.183    1.00    4881.   10225.
    ## 6 sigma.seasonal.7        0.0979 0.0109  0.0819  0.119    1.00    5081.   10211.

The *R̂*s and effective sample sizes look reasonable.

``` r
bayesplot::mcmc_trace(draws)
```

<img src="vax_files/figure-gfm/unnamed-chunk-2-1.png" width="672" />

Trace plots also look reasonable.

## Predictions

Now we’ll generate fits and predictions for a 180-day forecast:

``` r
forecast_days = 180

calc_fits = function(model = m, max_date = max(df_diff$date)) {
  df_log_diff %>%
    filter(date <= max_date) %>%
    add_draws(colSums(aperm(model$state.contributions, c(2, 1, 3))))
}
fits_all = calc_fits()

calc_pred_diff = function(model = m, max_date = max(df_diff$date)) {
  tibble(date = max_date + days(1:forecast_days)) %>%
    add_draws(predict(model, horizon = forecast_days)$distribution, value = "log_diff_vax")
}
pred_diff_all = calc_pred_diff()
```

Predictions from the model look like this (with imputed missing data in
gray):

``` r
x_dates = seq(make_date(2021, 1, 1), max(pred_diff_all$date), by = "month")
x_scale = function(...) list(
  scale_x_date(
    limits = range(df_diff$date, pred_diff_all$date),
    breaks = x_dates,
    labels = months(x_dates, abbreviate = TRUE)
  ),
  coord_cartesian(expand = FALSE, ...) 
)

summer_line = geom_vline(xintercept = make_date(2021, 9, 22), alpha = 0.25)

widths = c(.5, .8, .95)

plot_pred_diff = function(fits = fits_all, pred_diff = pred_diff_all) {
  pred_diff %>%
    ggplot(aes(date, log_diff_vax)) +
    stat_lineribbon(aes(y = .value, fill_ramp = stat(level)), data = fits, fill = "gray75", color = "gray65") +
    stat_lineribbon(.width = widths, color = "#08519c") +
    geom_line(data = df_log_diff, size = 1) +
    geom_vline(xintercept = min(pred_diff$date) - days(1), color = "gray75") +
    # summer_line +
    scale_fill_brewer() +
    theme_ggdist() +
    x_scale(ylim = c(-20, 5)) +
    labs(
      subtitle = "log(daily increase in logit(Percent vaccinated))",
      y = NULL,
      x = NULL
    )  
}
plot_pred_diff()
```

<img src="vax_files/figure-gfm/predictions_plot-1.png" width="682.56" />

There is quite a bit of uncertainty here, especially for far-out
forecasts.

We can translate these predictions of differences into predictions of
percent vaccinated by inverting the log, cumulatively summing
differences in log odds, then inverting the logit transformation:

``` r
calc_pred_vax = function(pred_diff = pred_diff_all) {
  last_observed_vax = df_diff %>%
    filter(date == min(pred_diff$date) - days(1)) %$%
    vax
  
  pred_diff %>%
    group_by(.draw) %>%
    mutate(
      vax = plogis(cumsum(c(
        qlogis(last_observed_vax) + exp(log_diff_vax[[1]]),
        exp(log_diff_vax[-1])
      )))
    )
}
pred_vax_all = calc_pred_vax()
```

Now we can plot the latent model alongside predictions of vaccination
rate and the predicted probability that the proportion of vaccinated
people is above some threshold (say, 70%):

``` r
plot_vaxed = function(pred_vax = pred_vax_all) {
  pred_vax %>%
    ggplot(aes(date, vax)) +
    stat_lineribbon(.width = widths, color = "#08519c") +
    scale_fill_brewer() +
    geom_line(data = df_diff, size = 1) +
    geom_hline(yintercept = .7, alpha = 0.25) +
    geom_vline(xintercept = min(pred_vax$date) - days(1), color = "gray75") +
    # summer_line +
    scale_y_continuous(limits = c(0,1), labels = scales::percent_format()) +
    x_scale() +
    theme_ggdist() +
    annotate("text", x = make_date(2020, 12, 28), y = 0.72, label = "70%", hjust = 0, vjust = 0, size = 3.25) +
    # annotate("text", x = make_date(2021, 9, 19), y = 0.25, hjust = 1, vjust = 0.5, label = "Sept 22\n End of summer", size = 3.25, lineheight = 1.05) +
    labs(
      subtitle = "Percent vaccinated (at least one dose)",
      y = NULL,
      x = NULL
    )
}

vaxed = plot_vaxed()

diffs = plot_pred_diff()

prob = pred_vax_all %>%
  group_by(date) %>%
  summarise(prob_vax_gt_70 = mean(vax > .70)) %>%
  ggplot(aes(date, prob_vax_gt_70)) +
  geom_line(color = "#08519c", size = 1) +
  theme_ggdist() +
  geom_hline(yintercept = 0.5, alpha = 0.25) +
  # summer_line +
  scale_y_continuous(limits = c(0,1), labels = scales::percent_format()) +
  x_scale() +
  labs(
    subtitle = "Pr(Percent vaccinated > 70%)",
    y = NULL,
    x = NULL
  )

vaxed / diffs / prob
```

<img src="vax_files/figure-gfm/model_breakdown-1.png" width="480" />

This gives us a final chart like this:

``` r
vaxed +
  stat_smooth(formula = y ~ x, method = lm, se = FALSE,
    data = df %>% filter(row_number() > n() - 8), 
    fullrange = TRUE, color = scales::alpha("white", 0.5), size = 1
  ) +
  annotate("text", 
    label = "white line=\nlinear model  ", 
    fontface = "bold", x = make_date(2021, 9, 12), y = .89, color = "black", hjust = 1, size = 3.25, lineheight = 1.05
  ) +
  labs(
    subtitle = "Forecasted % US with at least one dose, from time series model of daily increase in log odds"
  )
```

<img src="vax_files/figure-gfm/vaxed_plot-1.png" width="682.56" />

Anyway, my conclusion from all of this is essentially that there is *a
lot* of uncertainty in what the vaccination rate will be, at least if we
just look at the raw numbers, and doubtless the model I’ve shown here is
way oversimplified — but I have some trepidation about looking at even
simpler models (like linear projections) and ignoring their uncertainty,
as this is probably going to be at least a little misleading.

## Model comparison

For comparison purposes, let’s fit the model on data up to a few
different dates and see how it did on the data we’ve already observed.

``` r
plan(multisession)

models = tibble(
  upto = seq(as.Date("2021-03-01"), max(df$date), by = 15),
  model = future_map(upto, fit_model)
) 
```

Then build the charts for each model:

``` r
plots = map2(models$model, models$upto, function(m, max_date) {
  fits = calc_fits(m, max_date)
  pred_diff = calc_pred_diff(m, max_date)
  pred_vax = calc_pred_vax(pred_diff)
  list(
    vaxed = plot_vaxed(pred_vax),
    diffs = plot_pred_diff(fits, pred_diff)
  )
})

plots[[length(plots) + 1]] = list(
  vaxed = vaxed,
  diffs = diffs
)
```

And put them together into an animation:

``` r
png_path <- file.path("model_plots", "plot%03d.png")
png(png_path, type = "cairo", width = 1000, height = 1000, res = 200)
for (p in plots) {
  print(p$vaxed / p$diffs)
}
dev.off()
```

``` r
png_files <- sprintf(png_path, seq_along(plots))
gifski::gifski(png_files, "model_plots/animated.gif", width = 1000, height = 1000)
```

``` r
gganimate::gif_file("model_plots/animated.gif")
```

![](vax_files/figure-gfm/unnamed-chunk-7-1.gif)<!-- -->
