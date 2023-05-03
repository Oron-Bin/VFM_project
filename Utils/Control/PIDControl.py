
class PIDControl:
    def __init__(self, setpoint, feedback, output_lower, output_upper, kp, ki , kd):
        self.setpoint = setpoint #what i want to achieve
        self.feedback = feedback #what i get in real time
        self.output = 0
        self.output_lower = output_lower
        self.output_upper = output_upper
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.error_accumaltor = 0 #set the error to 0
        self.error = self.setpoint - self.feedback #defines the error

    def tune(self,kp,ki,kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def geterror(self):
        return self.setpoint - self.feedback

    def sample(self,setpoint,feedback):
        last_error = self.error
        self.setpoint = setpoint
        self.feedback = feedback
        self.error = self.setpoint - self.feedback
        self.error_accumaltor =  self.error_accumaltor + self.ki*self.error
        self.output = self  .kp*self.error + self.error_accumaltor - self.kd*(self.error-last_error)
        return self.clip(self.output)

    def clip(self, value):
        return(max(self.output_lower, min(value,self.output_upper)))

