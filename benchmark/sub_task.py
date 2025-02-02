from constants import *
from action import *

import omnigibson as og
from omnigibson.macros import gm
from omnigibson import object_states

# Make sure object states are enabled
gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = True
gm.USE_GPU_DYNAMICS = True

def pick_and_place(obj, pre_recep, is_on_floor, is_in, target_recep, target_pos, will_on_floor, robot, scene_model, \
                   trav_map=None, trav_map_img=None, trav_map_size=None, headless=False):
    """
    This function enables the robot to pick a object and place it to a target position
    obj: the object to pick [an omnigibson object]
    pre_recep: previous receptacle for the object to pick [an omnigibson object or None]
    is_on_floor: whether the object is on the floor, True for is and False for isn't [a bool]
    is_in: whether the object is in a receptacle, True for is and False for isn.t [a bool]
    target_recep: target receptacle for the object to place [an omnigibson object or None]
    target_pos: targe position for the object to place [a 3-array]
    will_on_floor: whether the object will be place on the floor, True for will and False for won't [a bool]
    robot: [fetch robot of omnigibson]
    scene_model: the name of the loaded scene model [a string]
    """

    paths = []

    # Find and pick the object
    if is_on_floor:
        paths.append(go_A(obj, robot, trav_map, trav_map_size))
    else:
        if "openable" in pre_recep.abilities:
            pre_recep.states[object_states.Open].set_value(new_value=False, fully=True)
        paths.append(go_A(pre_recep, robot, trav_map, trav_map_size))
    if is_in:
        open_A(pre_recep, robot)
        pick_A(obj, robot)
        close_A(pre_recep, robot, obj, "pick")
    else:
        pick_A(obj, robot)
    
    # Find the target place and place the object
    if will_on_floor:
        paths.append(go_P_with_B(target_pos, obj, robot, trav_map, trav_map_size))
        place_A_at_P(obj, target_pos, robot)
    else:
        if "openable" in target_recep.abilities:
            target_recep.states[object_states.Open].set_value(new_value=False, fully=True)
        paths.append(go_A_with_B(target_recep, obj, robot, trav_map, trav_map_size))
        if "openable" in target_recep.abilities:
            open_A(target_recep, robot, obj)
            place_A_at_P(obj, target_pos, robot, target_recep)
            close_A(target_recep, robot, obj, "place")
        else:
            place_A_at_P(obj, target_pos, robot, target_recep)
    
    wait_N(20, robot)

    return paths

def cook(obj, pre_recep, is_on_floor, is_in, cooker, target_recep, target_surface, robot, scene_model, \
         trav_map=None, trav_map_img=None, trav_map_size=None, headless=False):
    
    paths = []

    # Find and pick the object
    if is_on_floor:
        paths.append(go_A(obj, robot, trav_map, trav_map_size))
    else:
        if "openable" in pre_recep.abilities:
            pre_recep.states[object_states.Open].set_value(new_value=False, fully=True)
        paths.append(go_A(pre_recep, robot, trav_map, trav_map_size))
    if is_in:
        open_A(pre_recep, robot)
        pick_A(obj, robot)
        close_A(pre_recep, robot, obj, "pick")
    else:
        pick_A(obj, robot)

    # Cook the object
    if "openable" in cooker.abilities:
        cooker.states[object_states.Open].set_value(new_value=False, fully=True)
        paths.append(go_A_with_B(cooker, obj, robot, trav_map, trav_map_size))
        open_A(cooker, robot, obj)
        place_A_in_B(obj, cooker, robot)
        close_A(cooker, robot, obj, "place")
        set_A_at_P(obj, cooker.get_position())
        toggle_on_A(cooker, robot)
        wait_N(10, robot)
        cook_A(obj, robot)
        toggle_off_A(cooker, robot)
        open_A(cooker, robot)
        pick_A(obj, robot)
        close_A(cooker, robot, obj, "pick")
    else:
        paths.append(go_A_with_B(cooker, obj, robot, trav_map, trav_map_size))
        place_A_on_B(obj, cooker, robot)
        toggle_on_A(cooker, robot)
        wait_N(10, robot)
        cook_A(obj, robot)
        toggle_off_A(cooker, robot)
        pick_A(obj, robot)
    
    # Put the object
    paths.append(go_A_with_B(target_surface, obj, robot, trav_map, trav_map_size))
    place_A_on_B(obj, target_recep, robot)
    
    wait_N(10, robot)

    return paths

def pour(system, obj, pre_recep, is_on_floor, is_in, target_recep, target_surface, robot, scene_model, \
         trav_map=None, trav_map_img=None, trav_map_size=None, headless=False):
    
    paths = []

    # Find and pick the object
    if is_on_floor:
        paths.append(go_A(obj, robot, trav_map, trav_map_size))
    else:
        if "openable" in pre_recep.abilities:
            pre_recep.states[object_states.Open].set_value(new_value=False, fully=True)
        paths.append(go_A(pre_recep, robot, trav_map, trav_map_size))
    if is_in:
        open_A(pre_recep, robot)
        pick_A(obj, robot)
        close_A(pre_recep, robot, obj, "pick")
    else:
        pick_A(obj, robot)

    paths.append(go_A_with_B(target_surface, obj, robot, trav_map, trav_map_size))
    add_A_to_B_with_C(system, target_recep, obj, robot)

    # Place the object
    set_A_at_P(obj, target_recep.get_position() + np.array([1, 0, 0]))

    return paths

def wash(system, obj, pre_recep, is_on_floor, is_in, sink, robot, scene_model, \
         trav_map=None, trav_map_img=None, trav_map_size=None, headless=False):
    
    paths = []

    # Find and pick the object
    if is_on_floor:
        paths.append(go_A(obj, robot, trav_map, trav_map_size))
    else:
        if "openable" in pre_recep.abilities:
            pre_recep.states[object_states.Open].set_value(new_value=False, fully=True)
        paths.append(go_A(pre_recep, robot, trav_map, trav_map_size))
    if is_in:
        open_A(pre_recep, robot)
        pick_A(obj, robot)
        close_A(pre_recep, robot, obj, "pick")
    else:
        pick_A(obj, robot)

    # Clean the object
    paths.append(go_A_with_B(sink, obj, robot, trav_map, trav_map_size))
    place_A_at_P(obj, sink.states[object_states.ParticleSource].link.get_position() - np.array([0, 0, obj.native_bbox[2] * 0.5 + 0.1]), robot)
    add_water_to_A_with_B(obj, sink, robot)
    uncover_A_with_B(obj, system, robot)
    pick_A(obj, robot)
    
    wait_N(10, robot)

    return paths

def clean(system, obj, robot, scene_model, \
         trav_map=None, trav_map_img=None, trav_map_size=None, headless=False):
    
    paths = []

    paths.append(go_A(obj, robot, trav_map, trav_map_size))
    wait_N(50, robot)
    water = get_system("water")
    cover_A_with_B(obj, water, robot)
    uncover_A_with_B(obj, system, robot)
    wait_N(20, robot)
    uncover_A_with_B(obj, water, robot)
    
    wait_N(20, robot)

    return paths