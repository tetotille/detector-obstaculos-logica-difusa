# Detector de Obstáculos con Lógica Difusa

Este es un proyecto de trabajo final de grado (TFG) de la Facultad de Ingeniería de la Universidad Nacional de Asunción que busca realizar un detector de obstáculos utilizando algoritmos de lógica difusa para aplicarlo en barcos autotripulados que navegarán en el lago Ypakaraí y realizarán estudios de las aguas de forma automática.

## Instalación

La instalación se puede hacer de varias formas, las más fáciles son utilizando los scripts de instalación desarrollados para hacerlo en un paso. Sin embargo, en caso de tener dificultades en la utilización de dichos scripts también se especificará cómo hacerlo sin ellos.

### Powershell

Desde powershell ubicarse en la carpeta del proyecto y ejecutar el siguiente comando.

```powershell
powershell .\install.ps1
```

> Para poder ejecutar scripts de powershell es necesario habilitarlo siguiendo las indicaciones que aparecen al tratar de ejecutar el script.

### Fish

Desde el terminal Fish, podemos instalar ejecutando el siguiente comando una vez ubicados en la carpeta del proyecto.

```bash
fish ./install.fish
```

### Sin scripts automáticos

#### Creación del entorno virtual

Primeramente hay que crear un entorno virtual de python, para ello se puede utilizar el python venv que viene instalado por defecto en el software de python o también se puede utilizar algún gestor de entornos como Anaconda.

##### Python venv

Para crear un entorno virtual con venv podemos ejecutar el siguiente comando:

```bash
python -m venv .tesis
```

> En algunos casos si utilizas Windows podrías necesitar usar el comando py en lugar de python.

##### Anaconda

En caso de querer utilizar anaconda se puede usar el comando

```bash
conda create -n "tesis"
```

#### Activación del entorno virtual

Una vez creado el entorno virtual debemos de activarlo.

##### Python venv

Para activarlo depende del terminal que estés utilizando, puedes listar los scripts de activación con el siguiente comando:

```bash
ls .tesis/bin
# activate  activate.csh  activate.fish  Activate.ps1  f2py  pip  pip3  pip3.10  python  python3  python3.10
```

Según el terminal que usemos tendríamos que activar uno, en el caso de fish se utiliza `activate.fish` y en el caso de powershell se utiliza `Activate.ps1`. Con el siguiente comando.

```bash
source .tesis/bin/activate.fish # fish
```

```powershell
source .tesis\bin\Activate.ps1 # powershell
```

Una vez activato el entorno nos tiene que aparecer entre paréntesis el nombre de dicho entorno en nuestro terminal.

`(.tesis) tille@tille-82h8 ~/D/T/code (main)>`

##### Conda

Con el siguiente comando:

```bash
conda activate tesis
```

#### Instalación de requerimientos

Una vez activado el entorno podemos instalar las bibliotecas de python dependientes, esto se puede realizar en un comando.

Desde el terminal en la carpeta del proyecto ejecuta el siguiente comando.

```bash
pip install -r ./requirements.txt
```

De esta forma se instalarán todos los requisitos.

## Documentación

[Link a la Documentación](https://github.com/tetotille/detector-obstaculos-logica-difusa/blob/main/docs/docs.md)

## Roadmap

- Refactorizar el código para hacerlo escalable

- Implementar otros Detectores de obstáculos

- Implementar lógica difusa a los detectores más prometedores

- Implementar una lógica difusa general que se alimente de los demás algoritmos

## Autores

- [Liz Ozorio](https://github.com/liznoelia97)
  
- [Jorge Tillería](https://github.com/tetotille)

- Laboratorio de Sistemas Distribuidos de la Facultad de Ingeniería de la Universidad Nacional de Asunción