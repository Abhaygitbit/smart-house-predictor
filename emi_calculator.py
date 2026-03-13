def calculate_emi(principal, annual_rate, tenure_years):
    """
    Standard EMI Formula:
    EMI = P × r × (1+r)^n / ((1+r)^n - 1)

    P = Principal loan amount
    r = Monthly interest rate (annual_rate / 12 / 100)
    n = Total number of months (tenure_years * 12)
    """
    if annual_rate <= 0:
        n = tenure_years * 12
        monthly_emi = principal / n
        total_payment = principal
        total_interest = 0
        return round(monthly_emi, 2), round(total_payment, 2), round(total_interest, 2)

    r = annual_rate / (12 * 100)
    n = tenure_years * 12

    emi = principal * r * ((1 + r) ** n) / (((1 + r) ** n) - 1)
    total_payment = emi * n
    total_interest = total_payment - principal

    return round(emi, 2), round(total_payment, 2), round(total_interest, 2)
