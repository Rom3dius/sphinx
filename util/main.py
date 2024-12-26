import click
from typing import Dict

def parse_dict(input_string: str) -> Dict[str, str]:
    result = {}
    try:
        pairs = input_string.split(',')
        for pair in pairs:
            key_value = pair.split(':')
            if len(key_value) != 2:
                raise ValueError("Invalid input format")
            key, value = key_value
            result[key.strip()] = value.strip()
        return result
    except Exception as e:
        raise click.UsageError(f"Error parsing input: {str(e)}")

@click.group()
def cli():
    pass

@cli.command()
@click.argument('delay', type=int)
@click.argument('pathin', type=click.Path(exists=True, dir_okay=False))
@click.argument('pathout', type=click.Path(exists=True, file_okay=False))
def extract(delay, pathin, pathout):
    """ Saves frames every X frames from a video """
    import lib.cv as cv
    click.echo(f'Extracting a frame from {pathin} every {delay} milliseconds to {pathout}!')
    cv.extractImages(delay, pathin, pathout)
    click.echo("Done!")

# @cli.command()
# @click.argument('input_dict', type=parse_dict, required=True)
# def createdinodata(input_dict: Dict[str, str]):
#     """ Obligatory input dictionary in the format 'key1:value1,key2:value2,...' """
#     import lib.autodistill as ad
#     click.echo(f"Parsed dictionary: {input_dict}")
#     ad.create_dino_training_data(input_dict)

@cli.command()
@click.argument('dataset', type=click.Path(exists=True, file_okay=False))
def createseggptdata(dataset):
    """ Obligatory import of a labeled dataset """
    import lib.autodistill as ad
    click.echo(f"Loading seggpt...")
    ad.create_seggpt_training_data(dataset)

# @cli.command()
# @click.argument('input_dict', type=parse_dict, required=True)
# def createvlpartdata(input_dict: Dict[str, str]):
#     """ Obligatory input dictionary in the format 'key1:value1,key2:value2,...' """
#     import lib.autodistill as ad
#     click.echo(f"Parsed dictionary: {input_dict}")
#     ad.create_seggpt_training_data(input_dict)

@cli.command()
@click.argument('input_dict', type=parse_dict, required=True)
def createowldata(input_dict: Dict[str, str]):
    """ Obligatory input dictionary in the format 'key1:value1,key2:value2,...' """
    import lib.autodistill as ad
    click.echo(f"Parsed dictionary: {input_dict}")
    ad.create_owl_training_data(input_dict)

# @cli.command()
# @click.argument('input_dict', type=parse_dict, required=True)
# @click.argument('image', required=True, type=click.Path(exists=True, dir_okay=False, resolve_path=True))
# @click.option('-c', '--confidence', type=float, default=0.2)
# def testdino(input_dict: Dict[str, str], image: str, confidence: float):
#     """ Obligatory input dictionary in the format 'key1:value1,key2:value2,...' and path to an image"""
#     import lib.autodistill as ad
#     click.echo(f"Parsed dictionary: {input_dict}")
#     print(image)
#     ad.testdino(input_dict, image, confidence)

# @cli.command()
# @click.argument('input_dict', type=parse_dict, required=True)
# @click.argument('image', required=True, type=click.Path(exists=True, dir_okay=False, resolve_path=True))
# @click.option('-c', '--confidence', type=float, default=0.2)
# def testvlpart(input_dict: Dict[str, str], image: str, confidence: float):
#     """ Obligatory input dictionary in the format 'key1:value1,key2:value2,...' and path to an image"""
#     import lib.autodistill as ad
#     click.echo(f"Parsed dictionary: {input_dict}")
#     print(image)
#     ad.testvlpart(input_dict, image, confidence)

@cli.command()
@click.argument('input_dict', type=parse_dict, required=True)
@click.argument('image', required=True, type=click.Path(exists=True, dir_okay=False, resolve_path=True))
@click.option('-c', '--confidence', type=float, default=0.2)
def testowl(input_dict: Dict[str, str], image: str, confidence: float):
    """ Obligatory input dictionary in the format 'key1:value1,key2:value2,...' and path to an image"""
    import lib.autodistill as ad
    click.echo(f"Parsed dictionary: {input_dict}")
    print(image)
    ad.testowl(input_dict, image, confidence)

@cli.command()
@click.argument('dataset', type=click.Path(exists=True, file_okay=False))
@click.argument('confidence', type=float)
def filterconfidence(dataset: str, confidence: float):
    """ Remove low confidence values from dataset """
    import lib.autodistill as ad
    ad.filter_low_confidence_labels(dataset, confidence)

@cli.command()
@click.argument('datayaml', type=click.Path(exists=True))
def viewdataset(datayaml):
    """ Open app to view dataset, parameter is the data.yaml file """
    import lib.fiftyone as fo
    fo.launch(datayaml)

@cli.command()
def device():
    """ Check what device is in usage """
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Using device:', device)
    print()

    #Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

if __name__ == '__main__':
    cli()