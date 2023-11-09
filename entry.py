import controller

if __name__ == "__main__":
    controller.start()
    
    while controller.g_running:
        controller.update()