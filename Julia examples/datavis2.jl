using Gadfly
using Cairo

xvals = rand(1:5, 5)
yvals = rand(1:5, 5)

plot(x = xvals, y = yvals, Geom.point)

xvals2 = rand(0:100, 100)
yvals2 = rand(0:100, 100)

plot(x = sort(xvals2), y = sort(yvals2), Geom.point, Geom.line)

plot(x = xvals, y = yvals, Geom.point, Geom.smooth(method = :lm))

plot(layer(x = xvals2, y = yvals2, Geom.point), layer(x = sort(xvals2), y = sort(yvals2), Geom.point))

##Themes

plot(layer(x = xvals2, y= yvals2, Geom.point),
layer(x = sort(xvals2), y = sort(yvals2), Geom.point,
Theme(default_color=colorant"orange")))

points1 = layer(x = xvals2, y = yvals2, Geom.point, Theme(default_color=colorant"deepskyblue"))
points2 = layer(x = sort(xvals2), y = sort(yvals2), Geom.point, Theme(default_color=colorant"orange"))
plot(points1, points2, Guide.manual_color_key("Legend for this plot", ["Set of points", "Sorted points"],["deepskyblue", "orange"]))

plot(layer(x = xvals2, y=yvals2, Geom.point),
layer(x = sort(xvals2), y = sort(yvals2), Geom.point, Theme(default_color=colorant"orange")),
Theme(grid_color = colorant"white")
)

plot(points1, points2, Theme(grid_color=colorant"white", grid_color_focused = colorant"white"))

plot(x = xvals2, y = yvals2, Geom.point, Geom.smooth(method = :lm), Theme(line_width = 4px))

plot(layer(x=xvals2, y=yvals2, Geom.point, Theme(default_point_size = 4px)),
     layer(x = xvals2, y = yvals2, Geom.smooth(method = :lm), Theme(line_width=4px, default_color=colorant"orange")))


##Titles, Axis labels#######
plot(x=sort(xvals), y = sort(yvals), Geom.point,
Guide.title("My scatter plot"),
Guide.xlabel("x values"), Guide.ylabel("y values"))

using Distributions
plot(x=rand(Normal(), 100), y = Normal(), Stat.qq, Geom.point, Guide.title("QQ plot"))




##Density Plots####################################################
plot(x = xvals, Geom.density,
Guide.title("Dist"),
Theme(grid_color=colorant"white", line_width=3px))

plot(x = xvals2, Geom.histogram,
Guide.title("Variable 1 hist"))
