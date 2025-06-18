## ROS2-VLA_Bridge
This is the Project to the bachelor thesis of Samuel Rochlitzer in Computer Science at the Eberhardt Karls Universität Tübingen. It aims to implement an Interface to an external VLA in the ROS2 Framework.

---

### Summary Functions:

**1. ROS (panda3gpu):**
1.1 Access point, set prompt and parameters, start feedback loop  
1.2 Get image data  
1.3 Send request to VLA  
1.4 Get results from VLA  
1.5 Pass instructions on to controller

**2. VLA (Avalon):**
2.1 Access point, starting services (loop)  
2.2 Receiving data  
2.3 Passing data on to VLA  
2.4 Getting results from VLA  
2.5 Sending results back  

---

### Questions / Thoughts:

* What communication options should be considered / might already exist in ROS2?
* For coding and initial communication: SSH (tunneling)?
* Backend with Docker? With ROS2 as a node or as an independent application?
* One repo, two Docker containers?
* Adhere to Avalon best practices for environments, repositories, and data!
* Two packages?
* What nodes?
* What services / topics?

