import json
import os
import time

import airsim
import matplotlib.pyplot as plt


def save_image(image_data, filename):
    """Save image data to a file."""
    with open(filename, 'wb') as f:
        f.write(image_data)


def visualize_lidar_points(lidar_data):
    """Visualize Lidar points in 3D space."""
    if not lidar_data.points:
        print("No Lidar points received.")
        return

    x = [p[0] for p in lidar_data.points]
    y = [p[1] for p in lidar_data.points]
    z = [p[2] for p in lidar_data.points]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='b', marker='.')
    ax.set_title('Lidar Point Cloud')
    plt.show()


def test_cameras(client):
    """Test camera outputs."""
    print("Testing cameras...")
    responses = client.simGetImages([
        airsim.ImageRequest("front_center", airsim.ImageType.Scene),
        airsim.ImageRequest("thermal_view", airsim.ImageType.Thermal),
        airsim.ImageRequest("avoidance_view", airsim.ImageType.DepthPlanner)
    ])

    for i, response in enumerate(responses):
        if response and response.image_data_uint8:
            filename = f'image_{i}.png'
            save_image(response.image_data_uint8, filename)
            print(f"Saved camera image to {filename}")
        else:
            print(f"Camera {i} failed to capture data.")


def test_lidar(client):
    """Test Lidar sensor."""
    print("Testing Lidar...")
    lidar_data = client.getLidarData("Lidar")
    if lidar_data.points:
        print(f"Lidar received {len(lidar_data.points)} points.")
        visualize_lidar_points(lidar_data)
    else:
        print("No Lidar data received.")


def test_navigation(client):
    """Test basic drone navigation."""
    print("Testing navigation...")
    client.takeoffAsync().join()
    print("Drone has taken off.")

    client.moveToPositionAsync(10, 10, -10, 5).join()
    print("Drone moved to position (10, 10, -10).")

    client.hoverAsync().join()
    print("Drone is hovering.")

    client.landAsync().join()
    print("Drone has landed.")


def test_battery(client):
    """Test battery monitoring."""
    print("Testing battery status...")
    drone_state = client.getMultirotorState()
    if hasattr(drone_state, 'battery_voltage'):
        battery_level = drone_state.battery_voltage
        print(f"Battery Voltage: {battery_level} V")
    else:
        print("Battery monitoring not available.")


def test_collision_avoidance(client):
    """Test collision avoidance by moving toward an obstacle."""
    print("Testing collision avoidance...")
    client.moveToPositionAsync(0, 0, -2, 2).join()
    print("Collision avoidance test complete (check Unity for results).")


def run_tests():
    """Run all automated tests."""
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("Connected to AirSim.")

    # Create results directory
    results_dir = "AirSim_Test_Results"
    os.makedirs(results_dir, exist_ok=True)
    os.chdir(results_dir)

    # Test cameras
    test_cameras(client)

    # Test Lidar
    test_lidar(client)

    # Test navigation
    test_navigation(client)

    # Test battery
    test_battery(client)

    # Test collision avoidance
    test_collision_avoidance(client)

    print("All tests completed!")


if __name__ == "__main__":
    run_tests()
