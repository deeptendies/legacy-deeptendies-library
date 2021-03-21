if we want to access, best is to use gh in colab

##### terminal
to spin up a terminal, [there's several ways](https://stackoverflow.com/questions/59318692/how-can-i-run-shell-terminal-in-google-colab)
```
!bash
```
or
```
!pip install kora
from kora import console
console.start() 
```
or
if you have pro, just click the icon


##### gh cli

commands to install `gh`
```
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key C99B11DEB97541F0
sudo apt-add-repository https://cli.github.com/packages
sudo apt update
sudo apt install gh
```


##### auth
commands to authencitate, since it's cli it will fail, just go to this page, 
https://github.com/login/device
and from the example, paste the one time code that looks like this `C099-EAC9` to auth page.
```
/content# gh auth login
? What account do you want to log into? GitHub.com
? What is your preferred protocol for Git operations? HTTPS
? Authenticate Git with your GitHub credentials? No
? How would you like to authenticate GitHub CLI? Login with a web browser
! First copy your one-time code: C099-EAC9
- Press Enter to open github.com in your browser... 
/usr/bin/xdg-open: 851: /usr/bin/xdg-open: www-browser: not found
/usr/bin/xdg-open: 851: /usr/bin/xdg-open: links2: not found
/usr/bin/xdg-open: 851: /usr/bin/xdg-open: elinks: not found
/usr/bin/xdg-open: 851: /usr/bin/xdg-open: links: not found
/usr/bin/xdg-open: 851: /usr/bin/xdg-open: lynx: not found
/usr/bin/xdg-open: 851: /usr/bin/xdg-open: w3m: not found
xdg-open: no method available for opening 'https://github.com/login/device'
! Failed opening a web browser at https://github.com/login/device
  exit status 3
  Please try entering the URL in your browser manually
✓ Authentication complete. Press Enter to continue...
- gh config set -h github.com git_protocol https
✓ Configured git protocol
✓ Logged in as stancsz
```

then use cli as usual