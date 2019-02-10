module testcompile

prinln("does this even")


end



for col in 1:size(actualset,2)
    actual = actualset[:,col]
    firstdata = firstset[:,col]
    seconddata = secondset[:,col]

    mseone = (sum((firstdata - actual).^2))
    msetwo = (sum((seconddata - actual).^2))

    println(col)
    println(mseone, " ", msetwo, " : ", msetwo < mseone)
end



actualset = sae_results[1][3][1][1:30, 1]
firstset =  sae_results[1][3][2][1:30, 1]
secondset = sae_results[2][3][2][1:30, 1]

#Cumulative: Graph
actual_cumulative = cumsum(actualset)[end]
one_cumulative = cumsum(firstset)[end]
two_cumulative = cumsum(secondset)[end]

#gmseone = (one_cumulative - actual_cumulative).^2
#gmsetwo = (two_cumulative - actual_cumulative).^2
#gmseone < gmsetwo

#gmseone = abs(one_cumulative - actual_cumulative)
#gmsetwo = abs(two_cumulative - actual_cumulative)

gmseone = abs(sum(firstset) - sum(actualset))
gmsetwo = abs(sum(secondset) - sum(actualset))
gmseone < gmsetwo

#g1 = sum(firstset).^2 - 2*sum(firstset) * sum(actualset) + sum(actualset).^2
#g2 = sum(secondset).^2- 2*sum(secondset) * sum(actualset)+ sum(actualset).^2
#g1 < g2

#MSE epochs
#emseone = mean((firstset - actualset).^2)
#emsetwo = mean((secondset - actualset).^2)
#emseone < emsetwo

emseone = sum(abs.(firstset - actualset))
emsetwo = sum(abs.(secondset - actualset))
emseone < emsetwo

println(firstset)
e1 = sum(firstset.^2) -2*sum(firstset.*actualset)
e2 = sum(secondset.^2) -2*sum(secondset.*actualset)
e1 < e2
