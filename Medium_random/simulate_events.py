# Model a waiter(a process) who serves in a restaurant(an envt)

from collections import namedtuple
from random import random, seed
import simpy


def waiter(env):
    while True:
        print(f"Start taking orders from Customers at {env.now}")
        take_order_duration = 5
        yield env.timeout(take_order_duration)  # Models duration

        print(f"Start giving the orders to cooks at {env.now}")
        give_order_duration = 2
        yield env.timeout(give_order_duration)

        print(f"Start serving customers food at {env.now}\n")
        serve_order_duration = 5
        yield env.timeout(serve_order_duration)


env = simpy.Environment()  # Env where waiter lives
env.process(waiter(env))  # Pass the waiter to the env
env.run(until=30)  # Run simulation for 30 secs

# Simulate customers
# Model restaurant with a limited capacity


def customer(env, name, restaurant, **duration):
    while True:
        # There is a new customer between 0 and 10
        yield env.timeout(random()*10)
        print(
            f"{name} enters the restaurant and for the waiter to come at {round(env.now,2)}")
        with restaurant.request() as req:
            yield req

            print(
                f"Seats are available. {name} gets seated at {round(env.now, 2)}")
            yield env.timeout(duration['get_seated'])

            print(f"{name} starts looking at the menu at {round(env.now, 2)}")
            yield env.timeout(duration['choose_food'])

            print("Waiters start getting the order from {} at {}".format(
                name, round(env.now, 2)))
            yield env.timeout(duration['wait_for_food'])

            print("{} starts eating at {}".format(name, round(env.now, 2)))
            yield env.timeout(duration['eat'])

            print("{} starts paying at {}".format(name, round(env.now, 2)))
            yield env.timeout(duration['pay'])

            print("{} leaves at {}".format(name, round(env.now, 2)))


seed(1)
env = simpy.Environment()

# Model restaurant that can only allow 2 customers at once
restaurant = simpy.Resource(env, capacity=2)

durations = {'get_seated': 1, 'choose_food': 10,
             'give_order': 5, 'wait_for_food': 20, 'eat': 45, 'pay': 10}

for i in range(5):
    env.process(customer(env, f'Customer {i}', restaurant, **durations))

env.run(until=95)

# Put everything together, simulate a restaurant with a limited supply of food

NUM_ITEMS = 10  # number of items per food option

staff = simpy.Resource(env, capacity=1)
foods = ['Spicy Chicken', 'Poached Chicken', 'Tomato Chicken Skillet',
         'Honey Mustard Chicken']
available = {food: NUM_ITEMS for food in foods}
run_out = {food: env.event() for food in foods}
when_run_out = {food: None for food in foods}
rejected_customers = {food: 0 for food in foods}

Restaurant = namedtuple('Restaurant', 'staff, foods, available,'
                        'run_out, when_run_out, rejected_customers')
restaurant = Restaurant(staff, foods, available, run_out,
                        when_run_out, rejected_customers)

# Create a customer


def customer(env, food, num_food_order, restaurant):
    """Customer tries to order a certain number of a particular food, 
    if that food ran out, customer leaves. If there is enough food left,
    customer orders food."""

    with restaurant.staff.request() as customer:

        # If there is not enough food left, customer leaves
        if restaurant.available[food] < num_food_order:
            restaurant.rejected_customers[food] += 1
            return

        # If there is enough food left, customer orders food
        restaurant.available[food] -= num_food_order
        # The time it takes to prepare food
        yield env.timeout(10*num_food_order)

        # If there is no food left after customer orders, trigger run out event
        if restaurant.available[food] == 0:
            restaurant.run_out[food].succeed()
            restaurant.when_run_out[food] = env.now

        yield env.timeout(2)


def customer_arrivals(env, restaurant):
    """Create new customers until the simulation reaches the time limit"""
    while True:
        yield env.timeout(random.random()*10)

        # Choose a random food choice from the menu
        food = random.choice(restaurant.foods)

        # Number of a food choice the customer orders
        num_food_order = random.randint(1, 6)

        if restaurant.available[food]:
            env.process(customer(env, food, num_food_order, restaurant))


RANDOM_SEED = 20
SIM_TIME = 240

seed(RANDOM_SEED)
env = simpy.Environment()

# Start process and run
env.process(customer_arrivals(env, restaurant))
env.run(until=SIM_TIME)

for food in foods:
    if restaurant.run_out[food]:
        print(f'The {food} ran out {round(restaurant.when_run_out[food], 2)} '
              'minutes after the restaurant opens.')

        print(f'Number of people leaving queue when the {food} ran out is '
              f'{restaurant.rejected_customers[food]}.\n')
