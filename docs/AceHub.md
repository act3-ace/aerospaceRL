# Using Aerospace RL in ACEHUB

## Prerequisites
User must be logged into act3 VPN

## Setup ACE Profile
1.	Setup GitLab Token (To use ace users need a gitlab token)
    - Login to gitlab https://git.act3-ace.com
    - Enter user settings (select your avatar -> settings)
    - Select access tokens from tree view on the left
    - Create a new access token with all permissions
    - Save token for use on ACEHub 
2. Setup AceHub Profile: https://ace-hub.lynx.act3-ace.ai
    -	Login using AD (GitLab) credentials
    - Enter your profile by selecting user name at the top of screen (https://ace-hub.lynx.act3-ace.ai/profile)
    - In the Environment Variables section select the plus sign to add a varaibles
      - Set the Key to GIT_PASSWORD
      - Set the Value to the token created in Step 1
    - Save variables by selecting disc icon
    - In the Environment Files section select the plus sign to add a files
      - Set the Key to GIT_ASKPASS
      - Set the Value to:
        <code>
        #!/bin/sh
        echo $GIT_PASSWORD
        </code>
    - Save file by selecting disc icon
    - In the Image Pull Secrets Sections set the following values
      - Registry: reg.git.act3-ace.com
      - IPS Username: AD username
      - IPS Password: Gitlab token created in step 1

## Setting up AerospaceRL Environement
1. Create an Environment
    - Login to https://ace-hub.lynx.act3-ace.ai/environments
    - Select Create New Environment
    - In the Environment section set the following variables:
      - Name: to desired name (no space or symbols)
      - Image: reg.git.act3-ace.com/kyle.dunlap/aerospacerl:ace_hub
      - All Resource selected as desired
    - In the Environment Variable Section select + 
       - Set Key to FIXUID
       - Set Value to Act3 vpn userid (if unkown login to z-machine and run `id -u USERNAME`)
    - In the Environment Variable Section select + 
       - Set Key to FIXGID
       - Set Value to Act3 vpn userid (if unkown login to z-machine and run `id -u USERNAME`)
    - Select Save As Template to store Environment Parameters for future use

## Launching AerospaceRL Environement
1. Load Environment from the previous step
2. Select Launch 
    - this will load an environements screen and your environement will be `PENDING`
3. Enter UI 
    - When `PENDING` changes to `READY` click on environment name (this will should more info)
    - Select OPEN UI (this will open a brower link with vs code server)
4. To view tensorboard copy the vs code user and change then end of the url from `0-8888` to `0-6006`