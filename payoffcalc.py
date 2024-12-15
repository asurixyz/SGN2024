def calculate_payoffs(p1, p2, p3, p4):
    choices = [p1, p2, p3, p4]
    yes_count = sum(choices)
    no_count = 4 - yes_count
    if yes_count == 4:
        return "0 0 0 0"
    elif yes_count == 3 and no_count == 1:
        return f"{30 if p1 == 1 else -10} {30 if p2 == 1 else -10} {30 if p3 == 1 else -10} {30 if p4 == 1 else -10}"
    elif yes_count == 2 and no_count == 2:
        return f"{20 if p1 == 1 else -20} {20 if p2 == 1 else -20} {20 if p3 == 1 else -20} {20 if p4 == 1 else -20}"
    elif yes_count == 1 and no_count == 3:
        return f"{-10 if p1 == 1 else 30} {-10 if p2 == 1 else 30} {-10 if p3 == 1 else 30} {-10 if p4 == 1 else 30}"
    elif no_count == 4:
        return "0 0 0 0"
    else:
        return "0 0 0 0"

#Usage : 
payoffs = calculate_payoffs(0,0,1,0)
print(payoffs)
