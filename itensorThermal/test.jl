using ITensors

function main(; d1 = 2, d2 = 3)
  # ... your own code goes here ...
  # For example:
  i = Index(d1,"i")
  j = Index(d2,"j")
  T = random_itensor(i,j)
  @show T
end

main(; d1 = 4, d2 = 5)
