using Distributions
Pkg.add("Plots")
Pkg.add("PlotlyJS")
Pkg.add("StatPlots")
Pkg.add("HypothesisTests")
Pkg.add("GLM")
using Plots
using StatPlots
using HypothesisTests
using DataFrames
using GLM

plotlyjs()


a = [[0.5 0.9]; [0.1 0.4]]
heatmap(a,   colorscale= [["0.0", "rgb(165,0,38)"];["0.9", "rgb(215,48,39)"]])

norm1 = randn(1000);
histogram(norm1, bins=10, label="Std Normal Dist", title = "Hist")

mean(norm1)
std(norm1)

norm2 = rand(Normal(0, 1), 1000)
histogram(norm2, bins = 10, label = "Std Norm Dist", title = "hist2")
plot(Normal(0, 1), fill = (0.5, :blue), label ="Std Norm", title = "dens")

fit(Normal, norm1)

plot(Chisq(3), fill=(0.25, :blue), label = "3 dof")
plot!(Chisq(5), fill=(0.25, :blue), label = "5 dof")
plot!(Chisq(10), fill=(0.25, :blue), label = "10 dof")


##DataVis2############################################################

year1 = rand(Normal(67, 10), 100)
year2 = rand(Normal(71, 15), 100);
plot(Normal(67, 10), fill = (0.5, :orange), label = "Year 1", title = "Boxplot")
plot!(Normal(71, 15), fill = (0.5, :blue), label = "Year 2", title = "Boxplot")

skewness(year1), skewness(year2)
kurtosis(year1), kurtosis(year2)
EqualVarianceTTest(year1, year2)

data = DataFrame(One = year1, Two = year2)

OLS = glm(One, Two, data, Normal(), IdentityLink())

##Plotly###############################################################

plot(Plots.fakedata(50, 5), w = 1)
