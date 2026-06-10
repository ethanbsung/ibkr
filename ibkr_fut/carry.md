Strategy ten: Trade a portfolio of one or more instruments, each with positions scaled for a variable risk estimate. Scale positions according to the strength of their forecasted carry.

Carry is the component of excess returns for a future over and above what we earn form the spot price changing: Excess return = spot return + carry
Excess return (equities) = spot return + dividends - interest
Carry (equities) = divdends - interest

Excess return (bonds) = spot return + yield - repo rate
Carry (bonds) = yield - repo rate

Excess return (fx) = spot return + deposit rate - borrowing rate
Carry (fx) = deposit rate - borrowing rate

Excess return (STIR, Vol) = spot return + current spot price - futures price
Carry (STIR, Vol) = Current spot price - futures price

Excess return (Metals) = spot return - borrowing cost - storage costs
Carry (Metals) = -(borrowing cost + storage costs)

Excess return (Energy, Agricultural) = spot return - borrowing cost - storage costs + convenience yield
Carry (Energy, Agricultural) = -borrowing cost - storage costs + convienience yield

Measuring expected carry:
raw carry = price of nearer futures contract - price of currently held contract

If we assume that the gradient of the futures curve is constant, then the expected carry on the contract we are holding will be equal to the difference between the price of a farther out contract and the current contract:
raw carry = price of currently held contract - price of further out contract

This assumes we can get a price for a further out contract. Even if the second contract isn't liquid enough to be actively traded, there is normally a quoted price available for it. This price reflects the market's expectations about the various sources of carry. At the very least, there will be a price available for the further out contract shortly before the current contract is due to roll.

Annualization:
First we calculate the time between expiries as a fraction of a year. It's acceptable to approximate this using the difference in months:
expiry difference in years = abs(Months between contracts) / 12

Annualized raw carry = raw carry / expiry difference in years

Risk adjustment:
carry = annualized raw carry / (sigma_p * 16)
carry = annualized raw carry / (sigma_% * current contract price)

Carry as a trading strategy:
Since carry is defined as an expected annual return divided by an annualized standard deviation, both in the same price units, expected carry is equal to expected SR
Carry forecast = annualized raw carry / (sigma_p * 16)

Note on carry forecast:
For SP500, prior to 2000 there is a 'blocky' forecast because of sparse historical data for the second contract so carry can only be calculated on roll days. This is no longer a problem for SP500, but it is for many bond futures.

Carver prefers to trade a fixed month for seasonal commodities, where it's possible to do so


Actual content from book:


