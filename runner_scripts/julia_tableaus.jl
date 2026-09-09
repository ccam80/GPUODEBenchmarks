# Cash-Karp 5(4) and Fehlberg 4(5) tableaus for OrdinaryDiffEq's generic
# ExplicitRK stepper, entered as rationals (Cash & Karp 1990; Fehlberg, NASA TR
# R-315 1969) and built in the run's element type. Both propagate the 5th-order
# weights, matching cubie's cash-karp-54 / fehlberg-45 tableaus coefficient for
# coefficient. julia_algorithms.csv names them with `T` bound by julia_solver.

import DiffEqBase

function construct_cash_karp_54(::Type{T}) where {T}
    A = [
        0 0 0 0 0 0
        1//5 0 0 0 0 0
        3//40 9//40 0 0 0 0
        3//10 -9//10 6//5 0 0 0
        -11//54 5//2 -70//27 35//27 0 0
        1631//55296 175//512 575//13824 44275//110592 253//4096 0
    ]
    c = [0; 1 // 5; 3 // 10; 3 // 5; 1; 7 // 8]
    α = [37 // 378; 0; 250 // 621; 125 // 594; 0; 512 // 1771]
    αEEst = [2825 // 27648; 0; 18575 // 48384; 13525 // 55296; 277 // 14336; 1 // 4]
    return DiffEqBase.ExplicitRKTableau(map(T, A), map(T, c), map(T, α), 5;
        αEEst = map(T, αEEst), adaptiveorder = 4)
end

function construct_fehlberg_45(::Type{T}) where {T}
    A = [
        0 0 0 0 0 0
        1//4 0 0 0 0 0
        3//32 9//32 0 0 0 0
        1932//2197 -7200//2197 7296//2197 0 0 0
        439//216 -8 3680//513 -845//4104 0 0
        -8//27 2 -3544//2565 1859//4104 -11//40 0
    ]
    c = [0; 1 // 4; 3 // 8; 12 // 13; 1; 1 // 2]
    α = [16 // 135; 0; 6656 // 12825; 28561 // 56430; -9 // 50; 2 // 55]
    αEEst = [25 // 216; 0; 1408 // 2565; 2197 // 4104; -1 // 5; 0]
    return DiffEqBase.ExplicitRKTableau(map(T, A), map(T, c), map(T, α), 5;
        αEEst = map(T, αEEst), adaptiveorder = 4)
end
